// LLMs can sometimes work simultaneously on a task and have their outputs
// aggregated programmatically. This workflow, parallelization, manifests in
// two key variations:
//   Sectioning: Breaking a task into independent subtasks run in parallel.
//   Voting: Running the same task multiple times to get diverse outputs.
//
// When to use this workflow:
// Parallelization is effective when the divided subtasks can be parallelized
// for speed, or when multiple perspectives or attempts are needed for higher
// confidence results. For complex tasks with multiple considerations, LLMs
// generally perform better when each consideration is handled by a separate
// LLM call, allowing focused attention on each specific aspect.

import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0
})

const StateAnnotation = Annotation.Root({
  topic: Annotation<string>,
  joke: Annotation<string>,
  poem: Annotation<string>,
  story: Annotation<string>,
  combinedOutput: Annotation<string>
})

async function writeJoke(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(
    `Write a small joke about ${state.topic}`
  )
  return { joke: msg.content }
}

async function writePoem(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(
    `Write a small poem about ${state.topic}`
  )
  return { poem: msg.content }
}

async function writeStory(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(
    `Write a small story about ${state.topic}`
  )
  return { story: msg.content }
}

async function aggregator(state: typeof StateAnnotation.State) {
  const combined = `Here is a story, joke and poem about ${state.topic}\n` +
  `STORY:\n${state.story}\n\n` +
  `JOKE:\n${state.joke}\n\n` +
  `POEM:\n${state.poem}`;

  return { combinedOutput: combined }
}

export const graph = new StateGraph(StateAnnotation)
  .addNode("writeJoke", writeJoke)
  .addNode("writePoem", writePoem)
  .addNode("writeStory", writeStory)
  .addNode("aggregator", aggregator)
  .addEdge(START, "writeJoke")
  .addEdge(START, "writePoem")
  .addEdge(START, "writeStory")
  .addEdge("writeJoke", "aggregator")
  .addEdge("writePoem", "aggregator")
  .addEdge("writeStory", "aggregator")
  .addEdge("aggregator", END)
  .compile()