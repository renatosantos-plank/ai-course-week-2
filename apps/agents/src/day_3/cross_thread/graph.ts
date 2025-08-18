import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { InMemoryStore, LangGraphRunnableConfig, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph/web";
import { ChatOpenAI } from "@langchain/openai";
import { v4 } from "uuid";



const inMemoryStore = new InMemoryStore();
const llm = new ChatOpenAI({ model: 'gpt-4o-mini' })

const callModel = async (
  state: typeof MessagesAnnotation.State, 
  config: LangGraphRunnableConfig) => {
  
  const store = config.store
  if (!store) {
    throw new Error("Store is required when compiling the graph")
  }

  if (!config.configurable?.userId) {
    throw new Error("userId is required in the config")
  }

  const namespace = ["memories", config.configurable?.userId]
  const memories = await store.search(namespace)
  const info = memories.map(data => data.value.data).join("\n")
  const systemMsg = new SystemMessage(`You are a helpful assistant talking to the user. User info: ${info}`)
  
  const lastMessage = state.messages[state.messages.length - 1];
  if (
    typeof lastMessage.content === "string" &&
    lastMessage.content.toLocaleLowerCase().includes("remember")
  ) {
    await store.put(namespace, v4(), { data: lastMessage.content })
  }

  const response = await llm.invoke([
    systemMsg,
    ...state.messages
  ])

  return {
    messages: response
  }
}

export const graph = new StateGraph(MessagesAnnotation)
  .addNode("callModel", callModel)
  .addEdge(START, "callModel")
  .compile({
    store: inMemoryStore
  })

let config = { configurable: { thread_id: "1", userId: "1"} }
let inputMessage = new HumanMessage(`Hi! Remember: my name is Bob`)

for await (const chunk of await graph.stream({ messages: [ inputMessage ] }, { ...config, streamMode: "values"})) {
  // console.log(chunk.messages[chunk.messages.length - 1])
}

config = { configurable: { thread_id: "2", userId: "1" } };
inputMessage = new HumanMessage(`What is my name?`)
for await (const chunk of await graph.stream(
  { messages: [inputMessage] },
  { ...config, streamMode: "values" }
)) {
  // console.log(chunk.messages[chunk.messages.length - 1]);
}

const memories = await inMemoryStore.search(["memories", "1"]);
// console.log("memories", memories)
for (const memory of memories) {
    // console.log(await memory.value);
}