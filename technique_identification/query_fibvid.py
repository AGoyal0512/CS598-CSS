import pandas as pd
import cohere

def load_cohere_api_token():
    with open("cohere_api_key.txt", "r") as f:
        return f.read().strip()
    
API_TOKEN = load_cohere_api_token()

system_prompt = \
f"""You are acting as a moderator to annotate piece of misinformation with the psychological manipulation techniques used in them. Here are the definitions of the six psychological manipulation techniques:
- "polarization": The use of extreme or divisive language to polarize people into opposing groups.
- "conspiracy": The promotion of a false or misleading narrative about a public figure, event, or organization.
- "emotion": The use of emotionally charged language to manipulate people's feelings.
- "trolling": The deliberate attempt to provoke anger or annoyance in others.
- "impersonation": The use of false or misleading information to impersonate someone or something.
- "discredit": The attempt to discredit or damage the reputation of a public figure, event, or organization.
"""

user_prompt = \
"""Here is a headline from a news article, which is misinformation. Can you identify which of the following six psychological manipulation techniques are present in the claim? 
The six psychological manipulation techniques are: "polarization", "conspiracy", "emotion", "trolling", "impersonation", and "discredit".

### Statement:
{}

Return a comma-separated list of the manipulation techniques used in the claim.

### Your Response: 
"""

prompt = system_prompt + "\n\n" + user_prompt

co = cohere.ClientV2(API_TOKEN)
df = pd.read_csv(f"datasets/fibvid/claim_propagation.csv")

outputs_list = []

import time
import random

for statement in df["post_text"]:
    formatted_prompt = prompt.format(
        statement
    )
    
    max_retries = 5
    base_delay = 1
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model="command-r-plus-08-2024",
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature=0.0,
                max_tokens=50
            )
            outputs_list.append(response.message.content[0].text)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = (base_delay * 2 ** attempt) + (random.randint(0, 1000) / 1000)
            print(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)

df['Cohere-Command-R-Plus'] = outputs_list
df.to_csv(f"technique_identification/fibvid_evaluation_multiple.csv", index=False)
