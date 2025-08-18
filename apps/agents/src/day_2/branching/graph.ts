import { Annotation, END, START, StateGraph } from "@langchain/langgraph";


const StateAnnotation = Annotation.Root({
  aggregate: Annotation<string[]>({
    reducer: (a, b) => a.concat(b)
  }),
  which: Annotation<string>({
    reducer: (a, b) => b ?? a
  })
})


async function nodeA(state: typeof StateAnnotation.State) {
  console.log(`Writting I am A to ${state.aggregate}`)
  return {
    aggregate: ['I am A']
  }
}

async function nodeB(state: typeof StateAnnotation.State) {
  console.log(`Writting I am B to ${state.aggregate}`)
  return {
    aggregate: ['I am B']
  }
}

async function nodeC(state: typeof StateAnnotation.State) {
  console.log(`Writting I am C to ${state.aggregate}`)
  return {
    aggregate: ['I am C']
  }
}

async function nodeD(state: typeof StateAnnotation.State) {
  console.log(`Writting I am D to ${state.aggregate}`)
  return {
    aggregate: ['I am D']
  }
}

async function nodeE(state: typeof StateAnnotation.State) {
  console.log(`Writting I am E to ${state.aggregate}`)
  return {
    aggregate: ['I am E']
  }
}

function routeBCorCD(state: typeof StateAnnotation.State) {
  if (state.which === "bc") {
    return ["b", "c"]
  }
  return ["c", "d"]
}


export const graph = new StateGraph(StateAnnotation)
  .addNode("a", nodeA)
  .addNode("b", nodeB)
  .addNode("c", nodeC)
  .addNode("d", nodeD)
  .addNode("e", nodeE)
  .addEdge(START, "a")
  .addConditionalEdges("a", routeBCorCD, ["b", "c", "d"])
  .addEdge("b", "e")
  .addEdge("c", "e")
  .addEdge("d", "e")
  .addEdge("e", END)
  .compile()