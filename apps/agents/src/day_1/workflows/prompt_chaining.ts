// Workflow: Prompt Chaining
// 
// Prompt chaining is a technique where a complex task is broken down into a series of smaller, sequential steps. Each step is handled by a separate LLM call, with the output of one step serving as the input for the next. This approach allows for programmatic validation or "gating" at any intermediate stage, ensuring the workflow remains on track and meets desired criteria.
//
// When to use prompt chaining:
// - Best suited for tasks that can be clearly divided into fixed, logical subtasks.
// - Useful when you want to improve accuracy by simplifying each LLM call, even if it increases overall latency.
//
// Example use cases:
// - Creating marketing copy, then translating it into another language.
// - Generating an outline for a document, validating the outline, and then writing the full document based on the approved outline.

import { BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
})

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>,
  topic: Annotation<string>,
  joke: Annotation<string>,
  improvedJoke: Annotation<string>,
  finalJoke: Annotation<string>
})

async function generateTopic(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke([
    new SystemMessage(
      `You are a helpful assistante that can generate a joke topic from the user's question. The topic should be a single sentence that is no more than 10 words.`
    ),
    state.messages[state.messages.length - 1]
  ])
  return { topic: msg.content }
}

async function generateJoke(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(`write a short joke about ${state.topic}`)
  return { joke: msg.content }
}

function checkPunchline(state: typeof StateAnnotation.State) {
  if (state.joke?.includes("?") || state.joke?.includes("!")) {
    return "Pass"
  }
  return "Fail"
}

async function improveJoke(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(
    `Make this joke funnier by adding wordplay: ${state.joke}`
  )
  return { improvedJoke: msg.content }
}

async function polishJoke(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(
    `Add a surprising twist to this joke: ${state.improvedJoke}`
  )
  return { finalJoke: msg.content }
}

function isUserAskingJoke(state: typeof StateAnnotation.State) {
  return state.messages[state.messages.length - 1].content.toString().includes("joke") ? "Valid" : "Invalid"
}

const workflow = new StateGraph(StateAnnotation)
  .addNode("generateTopic", generateTopic)
  .addNode("generateJoke", generateJoke)
  .addNode("improveJoke", improveJoke)
  .addNode("polishJoke", polishJoke)
  .addConditionalEdges(START, isUserAskingJoke, {
      Valid: "generateTopic",
      Invalid: END
  })
  .addEdge("generateTopic", "generateJoke")
  .addConditionalEdges("generateJoke", checkPunchline, {
    Pass: "improveJoke",
    Fail: END
  })
  .addEdge("improveJoke", "polishJoke")
  .addEdge("polishJoke", END)

export const graph = workflow.compile()

