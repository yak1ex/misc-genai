from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from functools import reduce
import json
from typing import Iterable

import fire
from ollama import Client
from ruamel.yaml import YAML


@dataclass
class Response:
    calls: int = 0
    valid_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def __add__(self, other):
        response = replace(self)
        response += other
        return response

    def __iadd__(self, other):
        self.calls += other.calls
        self.valid_calls += other.valid_calls
        self.prompts += other.prompts
        return self


@dataclass
class Evaluation:
    calls: float
    valid_calls: float
    preservation: float
    enhancement: float
    variation: float


def distance(before: str, after: str, penalty: Iterable[str] = ('insert', 'delete', 'replace')) -> float:
    def reducer(value: tuple[int, int], element: tuple[str, int, int, int, int]) -> tuple[int, int]:
        type, before_from, before_to, after_from, after_to = element
        count = before_to - before_from + after_to - after_from
        return value[0] + (0 if type in penalty else count), value[1] + count
    matcher = SequenceMatcher(None, before, after)
    reduced = reduce(reducer, matcher.get_opcodes(), (0, 0))
    return reduced[0] / reduced[1]


def average(sequence: Iterable[float]) -> float:
    def reducer(value: tuple[float, float], element: float) -> tuple[float, float]:
        return value[0] + element, value[1] + 1
    reduced = reduce(reducer, sequence, (0.0, 0.0))
    if reduced[1] == 0.0:
        return 0.0
    return reduced[0] / reduced[1]


def extract_response_native(response, tool_spec: dict, prompt: str) -> Response:
    response_data = Response()
    tool_calls = response.get('message').get('tool_calls', [])
    allowed_params = tool_spec.get('parameters', {}).get('properties', {}).keys()
    required_params = tool_spec.get('parameters', {}).get('required', [])
    for tool_call in tool_calls:
        response_data.calls += 1
        tool_call.get('function', {}).get('name', '')
        if tool_spec['name'] == tool_call.get('name', ''):
            arguments = tool_call.get('parameters', {}).keys()
            if (all(argument in allowed_params for argument in arguments) and
                    all(required_param in arguments for required_param in required_params)):
                response_data.valid_calls += 1
                response_data.prompts.append(tool_call.get('parameters', {}).get('prompts'))
    return response_data


def proc_eval_native(
        client: Client,
        model_name: str,
        system: str,
        call: str,
        tool_spec: dict,
        history_type: str,
        prompt: str,
        history: list[dict]) -> Response:
    response = client.chat(model=model_name, messages=[
        {
            'role': 'system',
            'content': system
        },
        *history,
        {
            'role': 'user',
            'content': prompt
        }
    ], tools=[{'type': 'function', 'function': {k: v for k, v in tool_spec if k != 'type'}}])
    return extract_response_native(response, tool_spec, prompt)


def check_tool_call(tool_spec: dict, tool_call: dict) -> Response:
    response = Response(calls=1)
    if tool_spec['name'] == tool_call.get('name', ''):
        arguments = tool_call.get('parameters', {}).keys()
        allowed_params = tool_spec.get('parameters', {}).get('properties', {}).keys()
        required_params = tool_spec.get('parameters', {}).get('required', [])
        if (all(argument in allowed_params for argument in arguments) and
                all(required_param in arguments for required_param in required_params)):
            response.valid_calls += 1
            response.prompts.append(tool_call.get('parameters', {}).get('prompts'))
    return response


def extract_response_nonnative(response, tool_spec: dict, prompt: str) -> Response:
    call_results = Response(calls=0, valid_calls=0, prompts=[])
    try:
        content = response.get('message').get('content', '')
        content = content[content.find("{"):content.rfind("}") + 1]
        if content:
            result = json.loads(content)
            tool_calls = result.get("tool_calls") or [result]
            for tool_call in tool_calls:
                call_results += check_tool_call(tool_spec, tool_call)
    except Exception:
        pass
    return call_results


def make_query_default(system: str, messages: list[dict], prompt: str) -> str:
    history_part = "\n".join(
        f"{message['role'].upper()}: \"\"\"{message['content']}\"\"\""
        for message in messages[::-1][:4]
    )
    return f"History:\n{history_part}\nQuery: {prompt}"


make_query = {
    'default': make_query_default,
}


def proc_eval_nonnative(
        client: Client,
        model_name: str,
        system: str,
        call: str,
        tool_spec: dict,
        query_history_type: str,
        prompt: str,
        history: list[dict]) -> Response:
    tools_function_calling_prompt = call.replace("{{TOOLS}}", json.dumps(tool_spec))
    query = make_query.get(query_history_type, make_query_default)(system, history, prompt)
    messages = [
        {"role": "system", "content": tools_function_calling_prompt},
        {"role": "user", "content": f"Query: {query}"},
    ]
    # FIXME: query_history_type
    response = client.chat(
        model=model_name,
        messages=messages,
        stream=False,
    )
    print(f"{messages=}\n{response=}")
    return extract_response_nonnative(response, tool_spec, prompt)


def proc_eval(
        client: Client,
        model_name: str,
        system: str,
        call: str,
        tool: dict,
        native: bool,
        history_type: str,
        prompt: str,
        history: list[dict]) -> Response:
    proc_eval_ = proc_eval_native if native else proc_eval_nonnative
    return proc_eval_(client, model_name, system, call, tool, history_type, prompt, history)


def collect(data: list[Response]) -> Evaluation:
    # FIXME: Not yet implemented
    # distance(prompt, prompt_in_response, ('delete', 'replace'))
    # distance(prompt, prompt_in_response, ('equal', 'delete'))
    # distance(prev_prompt_in_response, prompt_in_response)
    return Evaluation(calls=0, valid_calls=0, preservation=0, enhancement=0, variation=0)


def eval(
        model_name: str,
        yaml_file: str,
        prompt: str,
        history_key: str = 'default',
        system_template: str = 'default',
        call_template: str = 'default',
        tool_template: str = 'default',
        native: bool = False,
        history_proc_type: str = 'default',
        count: int = 10,
        url: str = 'http://localhost:11434') -> Evaluation:
    with open(yaml_file) as yaml_stream:
        yaml = YAML(typ='safe')
        data = yaml.load(yaml_stream)
        system = data.get('system', {}).get(system_template, '')
        call = data.get('call', {}).get(call_template, '')
        tool = data.get('tool', {}).get(tool_template, {})
        history = data.get('history', {}).get(history_key, [])
        client = Client(host=url)
        data = [proc_eval(
            client,
            model_name,
            system,
            call,
            tool,
            native,
            history_proc_type,
            prompt,
            history) for _ in range(count)]
        return collect(data)


if __name__ == "__main__":
    print(distance('foo, bar, zot', 'foo, bar, zot, qux, quux', ('delete', 'replace')))
    print(distance('foo, bar, zot', 'foo, bar, zot, qux, quux', ('equal', 'delete')))
    fire.Fire(eval)
