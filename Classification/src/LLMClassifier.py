from openai import AzureOpenAI
from dotenv import load_dotenv
import yaml
import os
from typing import cast
import time

class LLMClassifier():
    def __init__(self) -> None:
        """
        Constructor for LLM_Classifier.

        Loads the configuration from the config/config.yaml file and creates an
        instance of AzureOpenAI with the specified endpoint and API key.

        The following configuration parameters are used:
        - deployment_endpoint: the URL of the Azure OpenAI endpoint
        - chat_llm_deployment_name: the name of the LLM deployment
        - temperature: the temperature to use for the LLM

        The API key is expected to be stored in the environment variable APIKEY.
        """
        load_dotenv(override=True)
        with open("config/config.yaml", "r", encoding="UTF-8") as config_file:
            config = yaml.safe_load(config_file)
            endpoint = str(config["deployment_endpoint"])
            #api_type = str(config["deployment_type"])
            self.model_name = str(config["chat_llm_deployment_name"])
            self.temperature = float(config["temperature"])

        self.client = AzureOpenAI(
            azure_endpoint = endpoint, 
            api_key=os.environ["APIKEY"],  
            api_version="2024-08-01-preview"
        )
    
    def single_classification(
            self,
            message: list,
            topic_names: list
        ) -> int:
        """
        Classify a single text into one of the given topic names.

        This function classifies a single text into one of the given topic names
        using the LLM specified in the configuration.

        Parameters
        ----------
        text : str
            The text to be classified.
        topic_names : list
            The list of possible topic names.
        filename : str
            The filename of the text to be classified (used for error reporting).

        Returns
        -------
        int
            The index of the topic name in the list, or -1 if the classification
            failed.
        """
        try:
            start = time.time()
            result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=message,
                        temperature=self.temperature,
                        max_tokens=10,
                    )
            result = result.choices[0].message.content
            end = time.time()
            print("")
            self.time_for_api_call.append(end-start)
        except Exception as e:
            result = "ERROR"
            print(e)
        for topic in topic_names:
            if topic in result:
                result = result = topic_names.index(topic)
                break
        if isinstance(result, str):
            if result == "ERROR":
                result = -2
            else:
                # append result to text file
                with open("errors.txt", "a") as f:
                    f.write(f"{result}\n")
                result = -1
        return result

    def batch_classification(
            self,
            texts: list,
            topic_names: list,
            train_text: list,
            train_target: list
        ) -> None:
        """
        Classifies a batch of texts into their respective topics.

        This function iterates over a list of texts and their corresponding filenames,
        classifying each text into one of the provided topic names using the 
        `single_classification` method. The predicted topic indices are stored in the 
        instance variable `self.y_pred`.

        Parameters
        ----------
        texts : list
            The list of texts to be classified.
        topic_names : list
            The list of possible topic names.
        filenames : list
            The list of filenames corresponding to the texts, used for error reporting.

        Returns
        -------
        None
        """
        self.time_for_api_call = []
        messages = self.create_messages(
            texts=texts,
            topic_names=topic_names,
            train_text=train_text,
            train_target=train_target
        )

        y_pred = []
        i = 0
        for message in messages:
            # print progress bar
            print(f"\r{i}/{len(messages)-1}", end="")
            y_pred.append(
                self.single_classification(
                    message=message,
                    topic_names=topic_names
                )
            )
            i+=1
        print("")
        self.y_pred = y_pred

    def create_few_shot_prompt(self, texts: list, target: list, topic_names: list) -> list:
        examples_text = []
        k = 0
        for i in range(len(texts)):
            if list(target)[i] == k:
                examples_text.append(texts[i])
                k += 1

        few_shot_prompts = [
            {
                "role": "user",
                "content": f""" Here is the text to classify: {t}"""
            } for t in examples_text
        ]

        few_shot_responses = [
            {
                "role": "assistant",
                "content": f"""Topic: {topic}"""
            } for topic in topic_names
        ]

        few_shot_examples = []
        for prompt, response in zip(few_shot_prompts, few_shot_responses):
            few_shot_examples.append(prompt)
            few_shot_examples.append(response)
        return few_shot_examples

    def create_system_prompt(self, topic_names: list) -> str:
        system_content = f"""You are a classifier for unstructured text. Consider the text and assign it to one of the following topics.

    topics: {topic_names}

    Make sure to respond with the topic name only. Answer as brief as possible"""
        system_prompt = {"role": "system", "content": system_content}
        return system_prompt
    
    def create_messages(self, texts: list, topic_names: list, train_text: list, train_target: list):
        system_prompt = self.create_system_prompt(topic_names)
        few_shot_examples = self.create_few_shot_prompt(
            texts=train_text,
            target=train_target,
            topic_names=topic_names
        )
        prompts = [{"role": "user", "content": f"""Here is the text to classify: {text}"""} for text in texts]
        assistant_response = {"role": "assistant", "content": "Topic:"}
        messages = [[system_prompt] + few_shot_examples + [prompt] + [assistant_response] for prompt in prompts]
        return messages
