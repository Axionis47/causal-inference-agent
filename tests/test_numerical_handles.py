"""Render selected metadata values, without repairing ungrounded source choices."""
import pytest

from backend.numerical import render_handles, validate
from backend.responses import Reply


@pytest.fixture(autouse=True)
def isolated():
    yield

def test_selected_numeric_handle_renders_exact_display_value_then_validates():
    reply=Reply(text='Estimate <n3>, interval <n2> to <n1>.',numbers=[{'source':'n3','value':3.56},{'source':'n2','value':1.87},{'source':'n1','value':5.25}])
    rendered=render_handles(reply)
    assert rendered.text=='Estimate 3.56, interval 1.87 to 5.25.'
    assert reply.text.startswith('Estimate <n3>')
    validate(rendered,{'n3':{'value':3.55965979},'n2':{'value':1.86994256},'n1':{'value':5.24937701}})

@pytest.mark.parametrize('numbers',[[],[{'source':'n1','value':1},{'source':'n1','value':2}]])
def test_missing_or_ambiguous_metadata_is_not_guessed(numbers):
    with pytest.raises(ValueError,match='exactly one'):
        render_handles(Reply(text='Result <n1>.',numbers=numbers))

def test_rendering_never_makes_a_false_number_pass_evidence_validation():
    value=render_handles(Reply(text='Result <n1>.',numbers=[{'source':'n1','value':8}]))
    with pytest.raises(ValueError,match='does not match'):
        validate(value,{'n1':{'value':2}})


def test_each_request_literal_has_a_distinct_unambiguous_display_handle():
    from backend.chat.evidence import numeric_options
    aliases, options=numeric_options({'question':{'basis':'prespecified_user_request','value':'Compare 1 versus 0 with 95% confidence.'},'worded':{'basis':'source_statement','value':'One month later.'}})
    assert aliases=={'n1':'question','n2':'question','n3':'question'}
    assert [item['value'] for item in options.values()]==[0,1,95]
    assert all(len(item['supported_values'])==1 for item in options.values())
