// When to use this workflow: Routing works well for complex tasks
// where there are distinct categories that are better handled
// separately, and where classification can be handled accurately,
// either by an LLM or a more traditional classification
// model/algorithm.
//
// Examples where routing is useful:
//
// Directing different types of customer service queries (general
// questions, refund requests, technical support) into different
// downstream processes, prompts, and tools.
//
// Routing easy/common questions to smaller models like Claude 3.5
// Haiku and hard/unusual questions to more capable models like
// Claude 3.5 Sonnet to optimize cost and speed.

import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import z from "zod";



const StateAnnotation = Annotation.Root({
  input: Annotation<string>,
  decision: Annotation<string>,
  output: Annotation<string>
})

const routeSchema = z.object({
  step: z.enum(['poem', 'story', 'joke'])
    .describe("The next step in the routing process")
})

const llm = new ChatOpenAI({
  model: 'gpt-4o-mini',
  temperature: 0
})

const llmRouter = llm.withStructuredOutput(routeSchema);



async function callRouter(state: typeof StateAnnotation.State) {
  const res = await llmRouter.invoke([
    new SystemMessage(`Route the input to story, joke, or poem based on the user's request.`),
    new HumanMessage(state.input)
  ])
  return { decision: res.step }
}

async function storytellerExpert(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke([
    new SystemMessage(`You are a storyteller expert`),
    new HumanMessage(state.input)
  ])
  return { output: msg.content }
}

async function comedianExpert(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke([
    new SystemMessage(`You are a comedian expert`),
    new HumanMessage(state.input)
  ])
  return { output: msg.content }
}

async function poetExpert(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke([
    new SystemMessage(`You are an expert poet`),
    new HumanMessage(state.input)
  ])

  return { output: msg.content }
}

function routeDecision(state: typeof StateAnnotation.State) {
  if (state.decision === "story") {
    return "storytellerExpert"
  } else if (state.decision === "joke") {
    return "comedianExpert"
  } else if (state.decision === "poem") {
    return "poetExpert"
  }
  return END
}

export const graph = new StateGraph(StateAnnotation)
  .addNode("callRouter", callRouter)
  .addNode("storytellerExpert", storytellerExpert)
  .addNode("comedianExpert", comedianExpert)
  .addNode("poetExpert", poetExpert)
  .addEdge(START, "callRouter")
  .addConditionalEdges("callRouter", routeDecision, ["storytellerExpert", "comedianExpert", "poetExpert"])
  .addEdge("comedianExpert", END)
  .addEdge("poetExpert", END)
  .addEdge("storytellerExpert", END)