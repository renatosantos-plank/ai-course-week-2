import { Annotation, END, MessagesAnnotation, START, StateGraph } from '@langchain/langgraph/web'
import { billingSupport, handleRefund, initialSupport, technicalSupport } from './nodes.js'
import { MemorySaver } from '@langchain/langgraph-checkpoint'

const checkpointer = new MemorySaver();
export const StateAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  nextRepresentative: Annotation<string>({
    reducer: (x, y) => y ?? x
  }),
  refundAuthorized: Annotation<boolean>
})

const routingFromInitialSupport = async (state: typeof StateAnnotation.State) => {
  
  if (state.nextRepresentative === "BILLING") {
    return "billing_support"
  } else if (state.nextRepresentative === "TECHNICAL") {
    return "techinical_support"
  } else {
    return END
  }
}

const routeFromBilling = async (state: typeof StateAnnotation.State) => {
  console.log("--->", state.nextRepresentative)
  if (state.nextRepresentative === "REFUND") {
    return "handle_refund"
  } else {
    return END
  }
}

export const graph = new StateGraph(StateAnnotation)
.addNode("initial_support", initialSupport)
.addNode("billing_support", billingSupport)
.addNode("techinical_support", technicalSupport)
.addNode("handle_refund", handleRefund)
.addEdge(START, "initial_support")
.addConditionalEdges("initial_support", routingFromInitialSupport, ["billing_support", "techinical_support", END])
.addEdge("techinical_support", END)
.addConditionalEdges("billing_support", routeFromBilling, ["handle_refund", END])
.addEdge("handle_refund", END)
.compile({checkpointer})
