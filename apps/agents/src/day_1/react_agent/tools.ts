import { tool } from "@langchain/core/tools"
import { TavilySearch } from "@langchain/tavily"
import { z } from 'zod'


const calculatorSchema = z.object({
  operation: z.enum(["add", "subtract", "multiply", "divide"]),
  x: z.number(),
  y: z.number()
})

export const calculator = tool(
  ({operation, x, y}) => {
    switch (operation) {
      case "add":
        return x + y;
      case "subtract":
        return x - y;
      case "multiply":
        return x * y;
      case "divide":
        return x / y;
      default:
        throw new Error("Invalid operation")
    }
  }, {
    name: "calculator",
    description: "A tool to calculate the sum, subtraction, multiplication or division of two numbers",
    schema: calculatorSchema
  }
)

export const tavilyTool = new TavilySearch({
  tavilyApiKey: process.env.TAVILIY_API_KEY,
  maxResults: 5
})