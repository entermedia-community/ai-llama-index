import os, json
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai_like import OpenAILike

class QueryEvent(Event):
    question: str


class AnswerEvent(Event):
    question: str
    answer: str


class SubQuestionQueryEngine(Workflow):
    @step
    async def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        if hasattr(ev, "query"):
            await ctx.store.set("original_query", ev.query)
            print(f"Query is {await ctx.store.get('original_query')}")

        if hasattr(ev, "llm"):
            await ctx.store.set("llm", ev.llm)

        if hasattr(ev, "tools"):
            await ctx.store.set("tools", ev.tools)

        response = (await ctx.store.get("llm")).complete(
            f"""
            Given a user question, and a list of tools, output a list of
            relevant sub-questions, such that the answers to all the
            sub-questions put together will answer the question. Respond
            in pure JSON without any markdown, like this:
            {{
                "sub_questions": [
                    "What is the population of San Francisco?",
                    "What is the budget of San Francisco?",
                    "What is the GDP of San Francisco?"
                ]
            }}
            Here is the user question: {await ctx.store.get('original_query')}

            And here is the list of tools: {await ctx.store.get('tools')}
            """
        )

        print(f"Sub-questions are {response}")

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]

        await ctx.store.set("sub_question_count", len(sub_questions))

        for question in sub_questions:
            self.send_event(QueryEvent(question=question))

        return None

    @step
    async def sub_question(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:
        print(f"Sub-question is {ev.question}")

        agent = ReActAgent.from_tools(
            await ctx.store.get("tools"),
            llm=await ctx.store.get("llm"),
            verbose=True,
        )
        response = agent.chat(ev.question)

        return AnswerEvent(question=ev.question, answer=str(response))

    @step
    async def combine_answers(
        self, ctx: Context, ev: AnswerEvent
    ) -> StopEvent | None:
        ready = ctx.collect_events(
            ev, [AnswerEvent] * await ctx.store.get("sub_question_count")
        )
        if ready is None:
            return None

        answers = "\n\n".join(
            [
                f"Question: {event.question}: \n Answer: {event.answer}"
                for event in ready
            ]
        )

        prompt = f"""
            You are given an overall question that has been split into sub-questions,
            each of which has been answered. Combine the answers to all the sub-questions
            into a single answer to the original question.

            Original question: {await ctx.store.get('original_query')}

            Sub-questions and answers:
            {answers}
        """

        print(f"Final prompt is {prompt}")

        response = (await ctx.store.get("llm")).complete(prompt)

        print("Final response is", response)

        return StopEvent(result=str(response))