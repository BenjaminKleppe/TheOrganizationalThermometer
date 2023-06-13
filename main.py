import time
import openai
import random
import pandas as pd
from openai.error import RateLimitError, InvalidRequestError
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# enter your own OpenAI API Key
openai.api_key = ''


def get_prompt(message):
    return f"department: {message['Department']}, sentiment: {message['Sentiment']}, feedback: {message['Feedback']}."


def get_prompt_template(prompt):
    template = f"\"{prompt}\" This prompt belongs to: [MASK]"
    return template


def get_role_prompt():
    # List of activities for the R&D department
    activities_rd = [
        "assign request", "request advice", "write recipe", "make sample",
        "review sample", "save reference sample", "review recipe",
        "send sample"
    ]

    # List of activities for the Quality department
    activities_quality = [
        "step 1.1: receive approval request", "step 1.2: check database", "step 1.3: control intended use",
        "step 2: register content", "step 3: perform second review"
    ]

    # Prompt for the role of the assistant
    role_prompt = f""" 
You are an assistant that helps with mapping feedback messages to the most likely business activities. The sentiment of 
the messages is given beforehand, and each message is feedback on business process activities on a product development 
process of an animal food production company. You should map the feedback to the most likely correlating business 
process activities. The process is a product development process, where a client requests a new food mixture. The R&D 
department writes a recipe and makes a sample. The Quality department reviews the quality. The messages can be from 
either the R&D department, or the Quality department.

The activities for the R&D department are: {', '.join(activities_rd)}. The process is started when a request comes 
in. First, the request is assigned to an employee of R&D. Then, the employee can asks for advice when the 
request is not fully filled in, when the request is faulty, or when the client needs to specify anything. After doing 
so, a recipe is written and placed in the database ("write recipe"). Then, the sample is prepared by weighing and c
ombining different raw materials ("make sample"). The sample is then tested by tasting if its tasty ("review sample") on Fridays. 
The preceding three activities are the development steps. The reference 
sample is then stored ("save reference sample"). Afterwards, the recipe is checked by weighing the ingredients 
("review recipe") and send to Quality department (which initializes the Quality process). After doing so, the sample 
is delivered to the client ("send sample").

The activities for the Quality department are: {', '.join(activities_quality)}. Furthermore, the activities: 
"receive approval request", "check database", and "control intended use" are the first step. If 
there are any issues during step 1, the recipe is send back to R&D. The second step is: "register content" 
where a declaration of the ingredients and made and reviewed. The third step in the Quality process is "perform second 
review", where the recipe/approval is checked for the second time. The feedback of the Quality 
department should be mapped to the corresponding step, not the activity. You can map the messages of R&D ONLY to the 
activities of the R&D department. You can map the messages of the Quality department ONLY to steps of the Quality 
department (e.g.: "step 1", "step 2", "step 3", "step 1, step 2, step 3"). The activities are presented in a 
sequential manner. When you receive a message as input, you return the activity. You only return the output, 
not the explanation. Therefore, you only return: "activity". For example, the message: "the raw materials are 
depleted" should give the following output: "monster maken", as you need raw materials to make a sample. Please do 
NOT give additional output."""

    return role_prompt


def load_data(n_shots):
    # Load the data from the Excel file
    df = pd.read_excel("feedback_product_development.xlsx", sheet_name='Data').dropna().to_dict(orient='records')

    # Create prompts with labels and sentiments
    messages = [(get_prompt(message), message['Label'], message['Sentiment']) for message in df]

    # Format the messages according to the template presented by the literature (Gao et al., 2021)
    template_messages = [(get_prompt_template(message[0]), message[1], message[2]) for message in messages]

    # Shuffle prompts
    random.shuffle(template_messages)

    # Split prompts into training and testing sets
    train_templates = template_messages[:n_shots]
    test_templates = template_messages[n_shots:]

    # Create the template messages for training
    training_prompts = [
        [
            {"role": "user", "content": f"{prompt[0]}."},
            {"role": "assistant", "content": f"{prompt[1]}"}
        ]
        for prompt in train_templates
    ]
    train_prompts = [item for sublist in training_prompts for item in sublist]
    print(train_prompts)

    # Create the test prompts
    test_prompts = [
        [
            ({"role": "user", "content": prompt[0]}, f"{prompt[1]}")
        ]
        for prompt in test_templates
    ]
    testing_prompts = [item for sublist in test_prompts for item in sublist]

    # Load the role prompt
    role_prompt = get_role_prompt()

    # Create the few-shot prompt list for the conversation
    training_prompts = [
                           {"role": "assistant", "content": role_prompt},
                           {"role": "user",
                            "content": f"I will give you {n_shots} examples, each followed by the correct solution"}
                       ] + train_prompts + [
                           {"role": "user", "content": "This were the examples, now you will now receive the feedback"}
                       ]

    return training_prompts, testing_prompts


def apply_model(training_prompts, testing_prompts):
    output, conversation, y_true, y_pred = [], training_prompts, [], []

    # Pre-train and evaluate model by iterating through the prompts
    for prompt in testing_prompts:
        # Print the prompt
        print(f"\nPrompt: {prompt[0]['content']} \nCorrect solution: {prompt[1]}")

        # Append prompt to conversation
        conversation[-1] = prompt[0]

        try:
            # Call OpenAI API
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation,
            )
        except RateLimitError:
            print("RateLimitError")
            time.sleep(10)
            # Call OpenAI API
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation,
            )
        except InvalidRequestError:
            print("InvalidRequestError")
            print(prompt[0])
            print(conversation)

        # Retrieve and print answer
        answer = chat_completion.choices[0].message.content
        print(f"ChatGPT: {answer}")

        # Save prompt, correct label, and prediction
        output.append({"prompt": prompt[0]['content'], "correct": prompt[1], "prediction": answer})
        y_true.append(prompt[1])
        y_pred.append(answer)

        # Sleep to prevent the limit of OpenAI API
        time.sleep(1)

    # Return prompts with corresponding predictions of the model
    return output, y_true, y_pred


def cf(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred, labels=list(set(y_true)))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(y_true)))
    display.plot(xticks_rotation='vertical')
    display.plot()


def main():
    # Load the data
    training_prompts, testing_prompts = load_data(n_shots=16)

    # Apply the prompt engineering model to the data
    output, y_true, y_pred = apply_model(training_prompts, testing_prompts)

    # Save results to an Excel sheet
    pd.DataFrame(output).to_excel("output.xlsx")

    # Print confusion matrix
    cf(y_pred, y_true)


if __name__ == "__main__":
    main()