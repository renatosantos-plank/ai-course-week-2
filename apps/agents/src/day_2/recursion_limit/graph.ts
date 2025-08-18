import { Annotation, END, GraphRecursionError, START, StateGraph } from "@langchain/langgraph";




const State = Annotation.Root({
  aggregate: Annotation<string[]>({
    reducer: (a, b) => a.concat(b),
    default: () => []
  }),
})

const a = async (state: typeof State.State) => {
  console.log(`node a sees [ ${state.aggregate} ]`)
  return {
    aggregate: ["a"]
  }
}

const b = async (state: typeof State.State) => {
  console.log(`node b sees [ ${state.aggregate} ]`)

  return {
    aggregate: ["b"]
  }
}

const route = (state: typeof State.State) => {
  if (state.aggregate.length < 7) {
    return "b"
  }
  return END
}

export const graph = new StateGraph(State)
  .addNode("a", a)
  .addNode("b", b)
  .addEdge(START, "a")
  .addConditionalEdges("a", route)
  .addEdge("b", "a")
  .compile()


try {
  await graph.invoke({ aggregate: [] }, { recursionLimit: 4 })
} catch (error) {
  if (error instanceof GraphRecursionError) {
    console.log("Recursion error")
  } else {
    throw error
  }
}
