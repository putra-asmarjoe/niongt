import OpenAI from "openai";
import express from "express";
import url from 'url';
import bp  from "body-parser";

// Importing the dotenv module to access environment variables

import dotenv  from "dotenv";

const openai = new OpenAI({
  apiKey:'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',
  organization:'org-6m9tmCf6oc3Ijv9NqSrNYPu9',
});

// const assistant = await openai.beta.assistants.create({
//     name: "Math Tutor",
//     instructions:
//         "You are a personal math tutor. Write and run code to answer math questions.",
//     tools: [{ type: "code_interpreter" }],
//     model: "gpt-4-1106-preview",
// });

const thread = await openai.beta.threads.create();

const message = await openai.beta.threads.messages.create(thread.id, {
    role: "user",
    content: "I need to solve the equation `3x + 11 = 14`. Can you help me?",
});

const run = await openai.beta.threads.runs.create(thread.id, {
    assistant_id: 'asst_0xX9VNvcF7GI0s86XBV0D6l3',//assistant.id,
    instructions: "Help user read data.",    
});

console.log(run)


//chatArray[Acts like a storage]
const chatArray = [];

// Creating a new Express app
const app = express();

// Using body-parser middleware to parse incoming request bodies as JSON
app.use(bp.json());

// Using body-parser middleware to parse incoming request bodies as URL encoded data
app.use(bp.urlencoded({ extended: true }));

// Importing and setting up the OpenAI API client
 

// Defining an endpoint to handle incoming requests
app.get("/converse", async(req, res) => {
  // Extracting the user's message from the request body
 
  var url_parts = url.parse(req.url, true);
  var message = url_parts.query.ask;

  // Calling the OpenAI API to complete the message
  
  const chatCompletion = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [{"role": "user", "content": message}],
  }).then((response) => {
    // Save the user's message and the AI's response to the chatArrayid
    chatArray.push({  id: response.id });
    chatArray.push({ role: "user", content: message });
    chatArray.push({ role: "assistant", content: response.choices[0].message.content });
    
     
    
  console.log(response);
  console.log(message+' : '+response.choices[0].message.content);
    // Return the chatArray as a JSON response
    res.json(chatArray);
  });
  
});




// Defining an endpoint to handle incoming requests
app.post("/talks", async(req, res) => {
  // Extracting the user's message from the request body
 console.log(req.body);
  // var url_parts = url.parse(req.url, true);
  // var message = url_parts.query.ask;
  const message = req.body.ask;//'hai aja';
  // Calling the OpenAI API to complete the message
  
  const chatCompletion = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [{    "role": "system",    "content": "Saya hanya ingin jawaban berdasarkan data yang disediakan. jangan mencari jawaban diluar data yang saya berikan. tidak di benarkan untuk menjawab dengan data diluar data yang saya berikan."  },
    {"role": "user", "content": message}],
  }).then((response) => {
    // Save the user's message and the AI's response to the chatArrayid
    chatArray.push({  id: response.id });
    chatArray.push({ role: "user", content: message });
    chatArray.push({ role: "assistant", content: response.choices[0].message.content });
    
     
    
  console.log(response);
  console.log(message+' : '+response.choices[0].message.content);
    // Return the chatArray as a JSON response
    res.json(chatArray);
  });
  
});





// Starting the Express app and listening on port 3000
app.listen(3275, () => {
  console.log("Conversational AI assistant listening on port 3275!");
});

// setTimeout(() => {
//     checkStatusAndPrintMessages(thread.id, run.id)
// }, 10000 );