// In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks,
// delegates them to worker LLMs, and synthesizes their results.
//
// When to use this workflow:
// This workflow is well-suited for complex tasks where you can’t predict the subtasks needed
// (in coding, for example, the number of files that need to be changed and the nature of the
// change in each file likely depend on the task).
// Whereas it’s topographically similar, the key difference from parallelization is its flexibility—
// subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Annotation, END, START, Send, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import z from "zod";

const sectionSchema = z.object({
  name: z.string().describe("Name for this section of the report"),
  description: z.string().describe(
    "Brief overview of the main topics and concepts to be covered in this section"
  )
})

const sectionsSchema = z.object({
  sections: z.array(sectionSchema).describe("Sections of the report")
})

const llm = new ChatOpenAI({model: 'gpt-4o-mini', temperature: 0})
const llmPlanner = llm.withStructuredOutput(sectionsSchema);

 
const StateAnnotation = Annotation.Root({
  topic: Annotation<string>,
  sections: Annotation<Array<z.infer<typeof sectionSchema>>>,
  completedSections: Annotation<string[]>({
    default: () => [],
    reducer: (a, b) => a.concat(b)
  }),
  finalReport: Annotation<string>
})

const WorkerStateAnnotation = Annotation.Root({
  section: Annotation<z.infer<typeof sectionSchema>>,
  completedSections: Annotation<string[]>({
    default: () => [],
    reducer: (a, b) => a.concat(b)
  })
})

async function orchestrator(state: typeof StateAnnotation.State) {
  const reportSections = await llmPlanner.invoke([
    new SystemMessage("Generate a plan for the report"),
    new HumanMessage(`Here is the report topic: ${state.topic}`)
  ])
  return { sections: reportSections.sections }
}

async function llmCall(state: typeof WorkerStateAnnotation.State) {
  const section = await llm.invoke([
    new SystemMessage("Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."),
    new HumanMessage(`Here is the section name: ${state.section.name} and description: ${state.section.description}`)
  ])

  return { completedSections: [section.content] }
}

async function synthesizer(state: typeof StateAnnotation.State) {
  const completedSections = state.completedSections;
  const completedReportSections = completedSections.join("\n\n---\n\n")

  return { finalReport: completedReportSections }
}

function assignWorkers(state: typeof StateAnnotation.State) {
  return state.sections.map((section) => new Send("llmCall", { section }))
}

export const graph = new StateGraph(StateAnnotation)
  .addNode('orchestrator', orchestrator)
  .addNode('llmCall', llmCall)
  .addNode('synthesizer', synthesizer)
  .addEdge(START, "orchestrator")
  .addConditionalEdges(
    "orchestrator",
    assignWorkers,
    ["llmCall"]
  )
  .addEdge("llmCall", "synthesizer")
  .addEdge("synthesizer", END)
  .compile()