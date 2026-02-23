# How to use this RAG Teaching Assistant on your own data
## Step-1 --> Collect your Sample videos
Move all your video files to your video folder.

## Step-2 --> Convert to Mp3
Convert all the video to Mp3 by running video_to_mp3 file.

## Step-3 --> Convert Mp3 to json
Convert all the Mp3 files to json by running mp3_to_json file.

## Step-4 --> Convert the json files to vectors
Use the file Processing_json to convert the json files to a dataframe with embeddings and save it to a joblib pickle.

## Step-5 --> Prompt generation and feeding to LLM
Read the joblib file and load it into the memory. Then create a relevent prompt as per the user query and feed it to the LLM.

## Step-6 --> Access the information
After feeding the query to LLM , finally access or gather the result from it.

## Here are some sample images of Project -->

The Project Interface

![image alt](https://github.com/amansingh3037/RAG_Based_AI_Assistant/blob/master/RAG%20Image%201.png?raw=true)

After Query Execution 

![image alt](https://github.com/amansingh3037/RAG_Based_AI_Assistant/blob/master/RAG%20Image%202.png?raw=true)
