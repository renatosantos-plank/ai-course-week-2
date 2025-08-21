import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages"
import { StateAnnotation } from "./graph.js"
import { model } from "./model.js"
import { BILLING_SYSTEM_TEMPLATE, INITIAL_CATEGORIZATION_HUMAN_TEMPLATE, INITIAL_CATEGORIZATION_SYSTEM_TEMPLATE, INITIAL_SUPPORT_SYSTEM_TEMPLATE, TECHNICAL_SYSTEM_TEMPLATE } from "./prompts.js"
import * as z from 'zod'
import { NodeInterrupt } from "@langchain/langgraph/web"

const billingSchema = z.object({
  nextRepresentative: z.enum(["REFUND", "RESPOND"])
})

const initialSupportSchema = z.object({
  nextRepresentative: z.enum(["BILLING", "TECHNICAL", "RESPOND"])
})
export const initialSupport = async (state: typeof StateAnnotation.State) => {

  const supportResponse = await model.invoke([
    new SystemMessage(INITIAL_SUPPORT_SYSTEM_TEMPLATE),
    ...state.messages
  ])

  
  const modelStructured = model.withStructuredOutput(initialSupportSchema)
  const categorizationResponse = await modelStructured.invoke([
    new SystemMessage(INITIAL_CATEGORIZATION_SYSTEM_TEMPLATE),
    ...state.messages,
    new HumanMessage(INITIAL_CATEGORIZATION_HUMAN_TEMPLATE)
  ])

  return {
    messages: [supportResponse],
    nextRepresentative: categorizationResponse.nextRepresentative
  }
}

export const billingSupport = async (state: typeof StateAnnotation.State) => {
  let trimmedMessages = state.messages;
  if (trimmedMessages.at(-1)?.getType() === "ai") {
    trimmedMessages = trimmedMessages.slice(0, -1)
  }

  const billingResponse = await model.invoke([
    new SystemMessage(BILLING_SYSTEM_TEMPLATE),
    ...trimmedMessages
  ])

  const BILLING_CATEGORIZATION_HUMAN_TEMPLATE = `
    The following text is a response from a customer support representative.
    Extract whether they want to refund the user or not.
    Respond with a JSON object containing a single key called "nextRepresentative" with one of the following values:
    
    If they want to refund the user, respond only with the word "REFUND".
    Otherwise, respond only with the word "RESPOND".
    
    Here is the text:
    
    <text>
    ${billingResponse.content}
    </text>.
  `

  const modelStructured = model.withStructuredOutput(billingSchema)
  const categorizationOutput = await modelStructured.invoke([
    new SystemMessage(BILLING_SYSTEM_TEMPLATE),
    new HumanMessage(BILLING_CATEGORIZATION_HUMAN_TEMPLATE),
  ])

  console.log("categorizationOutput", categorizationOutput)
  return {
    messages: [billingResponse],
    nextRepresentative: categorizationOutput.nextRepresentative
  }



}

export const technicalSupport = async (state: typeof StateAnnotation.State) => {
  let trimmedMessages = state.messages;
  if (trimmedMessages.at(-1)?.getType() === "ai") {
    trimmedMessages = trimmedMessages.slice(0, -1)
  }
  
  const response = await model.invoke([
    new SystemMessage(TECHNICAL_SYSTEM_TEMPLATE),
    ...trimmedMessages
  ])

  return {
    messages: response
  }
}

export const handleRefund = async (state: typeof StateAnnotation.State) => {
  console.log("handleRefund")
  if (!state.refundAuthorized) {
    console.log("--- HUMAN AUTHORIZATION REQUIRED FOR REFUND ---")
    throw new NodeInterrupt("Human authorization required.")
  }
  return {
    messages: new AIMessage("Refund processed!")
  }
}

