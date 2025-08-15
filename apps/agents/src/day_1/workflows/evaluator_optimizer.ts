import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import z from "zod";

const StateAnnotation = Annotation.Root({
  joke: Annotation<string>,
  topic: Annotation<string>,
  feedback: Annotation<string>,
  funnyOrNot: Annotation<string>,
})

const feedbackSchema = z.object({
  grade: z.enum(["funny", "not funny"]).describe(
    "Decide if the joke is funny or not"
  ),
  feedback: z.string().describe(
    "If the joke is not funny, provide feedback on how to improve it"
  )
})

const llm = new ChatOpenAI({ model: 'gpt-4o-mini', temperature: 0 })
const llmEvaluator = llm.withStructuredOutput(feedbackSchema);

async function generator(state: typeof StateAnnotation.State) {
  let msg = null;
  if (state.feedback) {
    msg = await llm.invoke(`
      Write a joke about ${state.topic} but take into account the feedback ${state.feedback}
    `)
  } else {
    msg = await llm.invoke(`Write a joke about ${state.topic}`)
  }

  return { joke: msg.content } 
}

async function evaluator(state: typeof StateAnnotation.State) {
  const grade = await llmEvaluator.invoke(`Grade the joke ${state.joke}`)
  return { 
    funnyOrNot: grade.grade,
    feedback: grade.feedback
  }
}

function routeJoke(state: typeof StateAnnotation.State) {
  if (state.funnyOrNot === "funny") {
    return "Accepted"
  } else if (state.funnyOrNot === "not funny") {
    return "Rejected + Feedback"
  }
  return "Accepted"
}

export const graph = new StateGraph(StateAnnotation)
  .addNode("generator", generator)
  .addNode("evaluator", evaluator)
  .addEdge(START, "generator")
  .addEdge("generator", "evaluator")
  .addConditionalEdges(
    "evaluator",
    routeJoke,
    {
      "Accepted": END,
      "Rejected + Feedback": "generator",
    }
  )
  .compile()  
