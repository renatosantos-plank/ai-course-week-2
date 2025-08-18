import { Annotation, Command, START, StateGraph } from "@langchain/langgraph";


const State = Annotation.Root({
  foo: Annotation<string>,
})

async function nodeA(state: typeof State.State) {
  console.log("called A")
  const goto = Math.random() > .5 ? "b" : "c"
  console.log(goto)
  return new Command({
    update: { foo: "a" },
    goto: goto
  })
}

async function nodeB(state: typeof State.State) {
  console.log("called B")

  return {
    foo: state.foo + "|b"
  }
}

async function nodeC(state: typeof State.State) {
  console.log("called C")

  return { 
    foo: state.foo + "|c"
  }
}



export const graph = new StateGraph(State)
  .addNode("a", nodeA, { ends: ["b", "c"] })
  .addNode("b", nodeB)
  .addNode("c", nodeC)
  .addEdge(START, "a")
  .compile()

