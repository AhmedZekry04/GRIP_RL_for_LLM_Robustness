"""
OpenAI Batch API helpers shared by the tagging, DLM and KV stages (50%
cheaper than synchronous calls, 24h SLA). Nothing here spends budget until a
stage is run with `--submit`.
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .common import ROOT

load_dotenv(os.path.join(ROOT, '.env'))

MAX_BATCH_REQUESTS = 50_000  # OpenAI hard cap per batch

_client = None


def client():
    """Created lazily so the pure request-building/parsing code can be
    imported without an OPENAI_API_KEY."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def make_chat_request(custom_id, model, messages, response_format=None):
    body = {'model': model, 'messages': messages}
    if response_format is not None:
        body['response_format'] = response_format
    return {'custom_id': custom_id, 'method': 'POST', 'url': '/v1/chat/completions', 'body': body}


def write_batch_input(requests, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in requests:
            f.write(json.dumps(r) + '\n')
    return path


def submit_batch(input_path, description=None):
    with open(input_path, 'rb') as f:
        uploaded = client().files.create(file=f, purpose='batch')
    batch = client().batches.create(
        input_file_id=uploaded.id,
        endpoint='/v1/chat/completions',
        completion_window='24h',
        metadata={'description': description or os.path.basename(input_path)},
    )
    return batch.id


def submit_batch_chunked(requests, base_path, description):
    """Split `requests` into batches of at most MAX_BATCH_REQUESTS, write each
    to `<base_path>_partN.jsonl`, submit them all, and return the batch IDs."""
    base, ext = os.path.splitext(base_path)
    batch_ids = []
    for i in range(0, len(requests), MAX_BATCH_REQUESTS):
        part_n = i // MAX_BATCH_REQUESTS + 1
        path = write_batch_input(requests[i:i + MAX_BATCH_REQUESTS], f'{base}_part{part_n}{ext}')
        batch_ids.append(submit_batch(path, description=f'{description}-part{part_n}'))
    return batch_ids


def submit_or_dry_run(requests, base_path, description, submit, poll_hint):
    """Write the request file and, only if `submit`, send it to the Batch API."""
    write_batch_input(requests, base_path)
    print(f"Wrote {len(requests)} requests to {base_path}")
    if not submit:
        print("Dry run only (pass --submit to send this batch and spend budget).")
        return
    batch_ids = submit_batch_chunked(requests, base_path, description)
    print(f"Submitted {len(batch_ids)} batch(es): {','.join(batch_ids)}")
    print(f"Once complete (~24h), run: {poll_hint} {','.join(batch_ids)}")


def poll_batch(batch_id, interval_seconds=60, timeout_seconds=24 * 3600):
    """Block until the batch reaches a terminal state."""
    elapsed = 0
    while elapsed < timeout_seconds:
        batch = client().batches.retrieve(batch_id)
        if batch.status in ('completed', 'failed', 'expired', 'cancelled'):
            return batch
        time.sleep(interval_seconds)
        elapsed += interval_seconds
    raise TimeoutError(f"Batch {batch_id} did not complete within {timeout_seconds}s")


def _read_result_file(file_id):
    content = client().files.content(file_id).text
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def fetch_batch_results(batch):
    """Returns {custom_id: response_body}. Requests that failed at the API
    level live in the separate error file; they get an empty body so they
    flow through the normal parse-failure path instead of vanishing."""
    if batch.status != 'completed':
        raise RuntimeError(f"Batch {batch.id} not completed (status={batch.status})")
    results = {}
    if batch.output_file_id is not None:
        for row in _read_result_file(batch.output_file_id):
            results[row['custom_id']] = row.get('response', {}).get('body', {})
    if batch.error_file_id is not None:
        for row in _read_result_file(batch.error_file_id):
            results.setdefault(row['custom_id'], {})
    return results


def poll_and_fetch(batch_ids):
    """Poll every batch in `batch_ids` (comma-separated string or list) and merge their results."""
    if isinstance(batch_ids, str):
        batch_ids = batch_ids.split(',')
    merged = {}
    for batch_id in batch_ids:
        merged.update(fetch_batch_results(poll_batch(batch_id)))
    return merged


def message_content(body):
    """The assistant message text of a response body, or None for a refusal
    or a malformed/failed response."""
    try:
        return body['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        return None
