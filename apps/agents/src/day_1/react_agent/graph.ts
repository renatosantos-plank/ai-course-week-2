import { ChatOpenAI } from "@langchain/openai"
import { calculator, tavilyTool } from "./tools.js";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, SystemMessage } from "@langchain/core/messages";
import { SYSTEM_PROMPT } from "./prompt.js";

const agentTools = [calculator, tavilyTool]

const agentModel = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0
}).bindTools(agentTools)


function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
  const lastMessage = messages[messages.length - 1] as AIMessage;

  if (lastMessage.tool_calls?.length) {
    return "tools";
  }
  return "__end__";
}

async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await agentModel.invoke([new SystemMessage(SYSTEM_PROMPT), ...state.messages]);
  return { messages: [response] }
}

const toolNode = new ToolNode(agentTools);

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent")
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue)

export const graph = workflow.compile()




