import { ChatGroq } from "@langchain/groq";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph/web";
import { Document } from "@langchain/core/documents";

const llm = new ChatGroq({
	model: "llama-3.1-8b-instant",
	temperature: 0,
});

const embeddings = new OpenAIEmbeddings({
	model: "text-embedding-3-large",
});

const vectorStore = new MemoryVectorStore(embeddings);

// LOAD, SPLIT, EMBED, STORE

// LOAD
const cheerioLoader = new CheerioWebBaseLoader(
	"https://lilianweng.github.io/posts/2023-06-23-agent/",
	{ selector: "p" },
);

const docs = await cheerioLoader.load();

// SPLIT
const splitter = new RecursiveCharacterTextSplitter({
	chunkSize: 1000,
	chunkOverlap: 200,
});
const allSplits = await splitter.splitDocuments(docs);

// INDEX (EMBED) and STORE
await vectorStore.addDocuments(allSplits);

const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

const inputStateAnnotation = Annotation.Root({
	question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
	question: Annotation<string>,
	context: Annotation<Document[]>,
	answer: Annotation<string>,
});

const retrieve = async (state: typeof inputStateAnnotation.State) => {
	const retrievedDocs = await vectorStore.similaritySearch(state.question);
	return { context: retrievedDocs };
};

const generrate = async (state: typeof StateAnnotation.State) => {
	const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
	const messages = await promptTemplate.invoke({
		question: state.question,
		context: docsContent,
	});
	const response = await llm.invoke(messages);
	return { answer: response.content };
};

export const graph = new StateGraph(StateAnnotation)
	.addNode("retrieve", retrieve)
	.addNode("generate", generrate)
	.addEdge(START, "retrieve")
	.addEdge("retrieve", "generate")
	.addEdge("generate", END)
	.compile();

let inputs = { question: "hello?" };
const result = await graph.invoke(inputs);
console.log(result);
