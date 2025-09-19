import os
import dotenv
from typing import Any, Dict, List
from pydantic import BaseModel
from openai import OpenAI


class NaturalLanguageProgram(BaseModel):
    program: str
class Variations(BaseModel):
    variations: List[str]


dotenv.load_dotenv()
resolved_openai = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_TOKEN")
model = "gpt-5"
client = OpenAI(api_key=resolved_openai)

def generate_natural_language_program(input_program: str) -> NaturalLanguageProgram:
    prompt = f"""
    Convert the following natural language program into a better natural language program that is easy to follow step-by-step.
    Do not do complex variables etc, make it in plain language. Simple to understand.

    Requirements:
    - Each step must be mechanically executable without interpretation
    - Use explicit spatial references (e.g., "row 3", "columns 1-4")
    - Specify exact values and conditions (e.g., "value equals 2", not "non-zero")
    - Avoid pattern recognition or analysis (no "determine", "find pattern", "analyze")
    - Number each step (1, 2, 3...)

    Good step examples:
    - "Copy all cells with value 3 to the same positions in output"
    - "Shift each cell in row 2 one position right"
    - "For each 2x2 region with sum > 5: set all cells to 1"

    Bad step examples:
    - "Identify the pattern and continue it"
    - "Find the rule that governs the transformation"
    - "Determine which objects are important"

    Only provide the program steps, nothing else. Do not make it complex at all just simple steps.
    That a human could follow. No complex variables or anything. Keep simplcity and meaning please, no complex stuff.
    Do not drastically change the meaning of the program, just make it more clear and simple to follow.
    Do not drastically increase the number of steps, keep it concise, and short. 
    The shorter the better, but still clear and simple to follow. No complex arithmetic or anything.
    You should not include any code or implementation details, any extra commentary or explanation.


    Input program:
    {input_program}

    Improved program:
    """
    resp = client.responses.parse(

            model = model,
            input = prompt,
            text_format = NaturalLanguageProgram,
                reasoning={
        "effort": "low",
        }
    )
    return resp.output_parsed
def variates(input_program: str) -> Variations:
    prompt = f"""
    You will be given a natural language program. 
    Generate 3 variations of the program that achieve the same goal but with different approaches.
    Each variation should follow the same format as the input program.
    Just write the variations, nothing else.
    Example output:
    1) Create a 6 by 6 grid
    2) Take the input pattern and repeat it along the first row
    ...

    You should not include any code or implementation details, any extra commentary or explanation.
    Just a list of variations(programs), make the variations wildly different, but still achieve the same goal.
    Input program:
    {input_program}
    """
    resp = client.responses.parse(

            model = model,
            input = prompt,
            text_format = Variations,
                reasoning={
        "effort": "low",
        }
    )
    return resp.output_parsed


'''
Example usage
program = """
1) Create a 6 by 6 grid
2) Take the input pattern and repeat it along the first row 
3) Reflect the input 2 by 2 pattern vertically(that is along the the two vertical block swap) 
4) Repeat this pattern in the next available row 
5) For the last row available space, repeat the initial pattern along the row.
"""
nlp = generate_natural_language_program(program).program
print(nlp)
vars = variates(program).variations
for v in vars:
    print(v)

'''