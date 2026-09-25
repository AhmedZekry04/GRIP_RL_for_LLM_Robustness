"""
KV acceptance checks: math_verify answer equivalence, triple blind-solve
consensus, and the structural diff between original and resampled question.

math_verify's built-in timeout uses signal.alarm() on POSIX but a subprocess
on Windows, which can fail silently and make every comparison return False.
Its timeouts are therefore disabled (parsing_timeout=None /
timeout_seconds=None) and replaced by a daemon-thread timeout, so a
pathological input fails closed (not equivalent) instead of hanging.
"""

import re
import threading

from math_verify import parse, verify

_TIMEOUT_SECONDS = 10

_NESTED_EXPONENT_TOWER = re.compile(r'\^\s*\{[^{}]*\^')
_HUGE_LITERAL_EXPONENT = re.compile(r'\^\s*\{?\s*\d{7,}')
_HUGE_FACTORIAL_ARG = re.compile(r'\d{5,}\s*!')

_TOKEN_PATTERN = re.compile(r'[a-zA-Z]+|\d+(?:\.\d+)?|\S')

_ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
         'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
_TENS = [None, None, 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']


def _run_with_timeout(fn, timeout=_TIMEOUT_SECONDS):
    """Run fn on a daemon thread; return None if it does not finish in time.
    A thread pool would block on shutdown waiting for the stuck worker."""
    result = {}

    def runner():
        try:
            result['value'] = fn()
        except Exception as e:
            result['error'] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    if 'error' in result:
        raise result['error']
    return result.get('value')


def _to_boxed(answer):
    """math_verify only applies its LaTeX normalisation (e.g. \\dfrac) on the
    \\boxed{} extraction path; bare answers like `\\dfrac{1}{2}` or `S_3`
    otherwise fail to parse."""
    if answer is None:
        return None
    return answer if '\\boxed{' in answer else '\\boxed{' + answer + '}'


def _has_dangerous_exponent(text):
    """Answers such as `37^{4^{15}}` would make sympy materialise
    billion-digit integers (a memory blow-up no timeout can interrupt), so
    exponent towers, 7+ digit exponents and 5+ digit factorials are treated
    as unverifiable."""
    if text is None:
        return False
    return bool(
        _NESTED_EXPONENT_TOWER.search(text)
        or _HUGE_LITERAL_EXPONENT.search(text)
        or _HUGE_FACTORIAL_ARG.search(text)
    )


def answers_equivalent(answer_a, answer_b):
    if _has_dangerous_exponent(answer_a) or _has_dangerous_exponent(answer_b):
        return False

    def _compare():
        gold = parse(_to_boxed(answer_a), parsing_timeout=None)
        pred = parse(_to_boxed(answer_b), parsing_timeout=None)
        return bool(verify(gold, pred, timeout_seconds=None))

    try:
        return _run_with_timeout(_compare) or False
    except Exception:
        return False


def blind_solve_consensus(back_synthesized_answer, blind_solve_answers):
    """All three blind answers must agree pairwise, and with the
    back-synthesised answer. Returns (passed, reason or None)."""
    if len(blind_solve_answers) != 3 or any(a is None for a in blind_solve_answers):
        return False, 'missing blind-solve answer(s)'

    for i in range(3):
        for j in range(i + 1, 3):
            if not answers_equivalent(blind_solve_answers[i], blind_solve_answers[j]):
                return False, f'blind-solve answers {i} and {j} disagree'

    if not answers_equivalent(blind_solve_answers[0], back_synthesized_answer):
        return False, 'blind-solve consensus disagrees with back-synthesized answer'

    return True, None


def _replace_whole_number(text, value, placeholder):
    """Replace `value` only as a standalone number: not inside a longer
    number or decimal, but still before a sentence-ending period. A trailing
    ordinal suffix (21st -> 19th) is swallowed with it."""
    pattern = re.compile(rf'(?<!\w)(?<!\d\.){re.escape(str(value))}(?:st|nd|rd|th)?(?!\w)(?!\.\d)')
    return pattern.sub(placeholder, text)


def _number_to_words(n):
    if n < 0 or n >= 1000:
        return None
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f'-{_ONES[ones]}' if ones else '')
    hundreds, rest = divmod(n, 100)
    return f'{_ONES[hundreds]} hundred' + (f' {_number_to_words(rest)}' if rest else '')


def _word_forms(value):
    """Spelled-out forms of an integer 0-999 (`6` -> `six`), since questions
    often write small constants as words."""
    if not re.fullmatch(r'\d+', str(value)):
        return []
    words = _number_to_words(int(value))
    return [] if words is None else [words, words.replace('-', ' ')]


def _replace_word_form(text, word, placeholder):
    return re.sub(rf'(?<![a-zA-Z]){re.escape(word)}(?![a-zA-Z])', placeholder, text)


def structural_diff_ok(original_question, new_question, changed_constants):
    """True if the two questions are token-for-token identical once every
    old and new constant value (digit or word form) is masked out."""
    def strip_constants(text):
        for old, new in changed_constants:
            text = _replace_whole_number(text, old, '\0')
            text = _replace_whole_number(text, new, '\0')
            for word in _word_forms(old) + _word_forms(new):
                text = _replace_word_form(text, word, '\0')
        return [t for t in _TOKEN_PATTERN.findall(text) if t != '\0']

    return strip_constants(original_question) == strip_constants(new_question)
