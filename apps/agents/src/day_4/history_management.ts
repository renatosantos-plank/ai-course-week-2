// filterNode, with filterMessages
// trimNode, with TrimMessages
// deleterNode, with RemoveMessage

import { AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage, filterMessages, trimMessages } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { END, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph/web";
import { ChatOpenAI } from "@langchain/openai";

const checkpointer = new MemorySaver()
const llm = new ChatOpenAI({ model: 'gpt-4o-mini' })


async function rootNode(state: typeof MessagesAnnotation.State) {
  const testMessages = [
    new SystemMessage("you are a helpfull assistant"),
    new HumanMessage("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
    new AIMessage("Hi! I'm here to help you with your questions."),
    new HumanMessage("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
    new AIMessage("Hi! I'm here to help you with your questions."),
    new ToolMessage("Calculator result: 42", "calculator", "calc_123"),
    new HumanMessage("Can you help me with math?"),
    new AIMessage("Of course! I can help you with calculations."),
    new ToolMessage("Addition result: 15 + 27 = 42", "calculator", "calc_124"),
    new HumanMessage("What's the weather like?"),
    new AIMessage("I don't have access to real-time weather data, but I can help you with other questions!")
  ];
  
  return {
    messages: testMessages
  }
}

async function filterNode(state: typeof MessagesAnnotation.State) {
  // filtering messages that we want to remove
  const filteredMessages = filterMessages(state.messages, {
    // includeNames: ["human", "ai"],
    // includeIds
    includeTypes: ["tool"]
    // excludeNames
    // excludeIds
    // excludeTypes
  })

  // actually removing them from state
  const removeMessages = filteredMessages.map(msg => new RemoveMessage({id: msg.id || ""}))
  return { 
    messages: removeMessages
  }
}

async function trimNode(state: typeof MessagesAnnotation.State) {
  // trim messages to the max of tokens
  const trimmedMessages = await trimMessages(state.messages, {
    maxTokens: 50,
    strategy: "last",
    tokenCounter: new ChatOpenAI({ modelName: "gpt-4o-mini" }),
    startOn: "human",
    endOn: ["human", "tool"],
    includeSystem: true
  })

  // Find messages that were removed by comparing IDs
  const removedMessages = state.messages.filter(msg => 
    !trimmedMessages.some(trimmedMsg => trimmedMsg.id === msg.id)
  );
  console.log("messages to remove:", removedMessages.length);
  
  // Create RemoveMessage objects for the messages that were trimmed out
  const removeMessages = removedMessages.map(msg => new RemoveMessage({id: msg.id || ""}));
  
  return {
    messages: removeMessages
  };
}



export const graph = new StateGraph(MessagesAnnotation)
.addNode("root", rootNode)
.addNode("filter", filterNode)  
.addNode("trim", trimNode)
.addEdge(START, "root")
.addEdge("root", "filter")
.addEdge("filter", "trim")
.addEdge("trim", END)
.compile({ checkpointer })
  
const config = { configurable: { thread_id: "1" }}

const response = await graph.invoke({}, config)
// console.log("\n\nresponse")
// console.log(response)