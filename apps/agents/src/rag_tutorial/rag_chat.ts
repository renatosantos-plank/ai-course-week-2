import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { tool } from "@langchain/core/tools";
import { ChatGroq } from "@langchain/groq";
import {
	ToolNode,
	createReactAgent,
	toolsCondition,
} from "@langchain/langgraph/prebuilt";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import z from "zod";
import {
	END,
	MessagesAnnotation,
	START,
	StateGraph,
} from "@langchain/langgraph/web";
import {
	AIMessage,
	BaseMessage,
	HumanMessage,
	SystemMessage,
	ToolMessage,
	isAIMessage,
} from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { QdrantVectorStore } from "@langchain/qdrant";

const llm = new ChatGroq({ model: "llama-3.1-8b-instant", temperature: 0 });
const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-large" });

const vectorStore = new QdrantVectorStore(embeddings, {
	collectionName: "rag-documents",
	url: process.env.QDRANT_URL,
	apiKey: process.env.QDRANT_API_KEY,
});

const checkpointer = new MemorySaver();

const cheerioLoader = new CheerioWebBaseLoader(
	"https://lilianweng.github.io/posts/2023-06-23-agent/",
	{
		selector: "p",
	},
);

const docs = await cheerioLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
	chunkSize: 1000,
	chunkOverlap: 200,
});

const allSplits = await splitter.splitDocuments(docs);

await vectorStore.addDocuments(allSplits);

const retrieverSchema = z.object({
	query: z.string(),
});

const retrieveTool = tool(
	async ({ query }) => {
		const retrievedDocs = await vectorStore.similaritySearch(query, 2);
		const serialized = retrievedDocs
			.map(
				(doc) => `Source: ${doc.metadata.source}\nContent:  ${doc.pageContent}`,
			)
			.join("\n");

		return [serialized, retrievedDocs];
	},
	{
		name: "Retrieve",
		description: "Retrieve information related to a query",
		schema: retrieverSchema,
		responseFormat: "content_and_artifact", // ?
	},
);

const retrieveToolNode = new ToolNode([retrieveTool]);

const queryOrRespond = async (state: typeof MessagesAnnotation.State) => {
	const llmWithTools = llm.bindTools([retrieveTool]);
	const response = await llmWithTools.invoke(state.messages);
	return { messages: [response] };
};

const generate = async (state: typeof MessagesAnnotation.State) => {
	const recentToolMessages = [];
	for (let i = state["messages"].length - 1; i >= 0; i--) {
		const message = state["messages"][i];
		if (message instanceof ToolMessage) {
			recentToolMessages.push(message);
		} else {
			break;
		}
	}

	const toolMessages = recentToolMessages.reverse();

	const docsContent = toolMessages.map((doc) => doc.content).join("\n");

	const systemMessageContent =
		"You are an assistant for question-answering tasks. " +
		"Use the following pieces of retrieved context to answer " +
		"the question. If you don't know the answer, say that you " +
		"don't know. Use three sentences maximum and keep the " +
		"answer concise." +
		"\n\n" +
		`${docsContent}`;

	const conversationMessages = state.messages.filter(
		(message) =>
			message instanceof HumanMessage ||
			message instanceof SystemMessage ||
			(message instanceof AIMessage && message.tool_calls?.length === 0),
	);

	const prompt = [
		new SystemMessage(systemMessageContent),
		...conversationMessages,
	];

	const response = await llm.invoke(prompt);

	return { messages: [response] };
};

export const graph = new StateGraph(MessagesAnnotation)
	.addNode("queryOrRespond", queryOrRespond)
	.addNode("tools", retrieveToolNode)
	.addNode("generate", generate)
	.addEdge(START, "queryOrRespond")
	.addConditionalEdges("queryOrRespond", toolsCondition, {
		__end__: END,
		tools: "tools",
	})
	.addEdge("tools", "generate")
	.addEdge("generate", END)
	.compile({ checkpointer });

// const threadConfig = {
// 	configurable: { thread_id: "abc123" },
// 	streamMode: "values" as const,
// };

const prettyPrint = (message: BaseMessage) => {
	let txt = `[${message._getType()}]: ${message.content}`;
	if ((isAIMessage(message) && message.tool_calls?.length) || 0 > 0) {
		const tool_calls = (message as AIMessage)?.tool_calls
			?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
			.join("\n");
		txt += ` \nTools: \n${tool_calls}`;
	}
	console.log(txt);
};

// let inputs3 = {
// 	messages: [{ role: "user", content: "What is Task Decomposition?" }],
// };

// for await (const step of await graph.stream(inputs3, threadConfig)) {
// 	const lastMessage = step.messages[step.messages.length - 1];
// 	prettyPrint(lastMessage);
// 	console.log("-----\n");
// }

// let inputs4 = {
// 	messages: [
// 		{ role: "user", content: "Can you look up some common ways of doing it?" },
// 	],
// };

// for await (const step of await graph.stream(inputs4, threadConfig)) {
// 	const lastMessage = step.messages[step.messages.length - 1];
// 	prettyPrint(lastMessage);
// 	console.log("-----\n");
// }

const agent = createReactAgent({ llm: llm, tools: [retrieveTool] });

let inputMessage = `What is the standard method for Task Decomposition?
Once you get the answer, look up common extensions of that method.`;

let inputs5 = { messages: [{ role: "user", content: inputMessage }] };

for await (const step of await agent.stream(inputs5, {
	streamMode: "values",
})) {
	const lastMessage = step.messages[step.messages.length - 1];
	prettyPrint(lastMessage);
	console.log("-----\n");
}
