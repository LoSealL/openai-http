import openai_http


class MyBackend(openai_http.BackendBase):

    async def generate(self, prompt, **kwargs):
        text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        return {
            "generated_text": f"Echo: {text}",
            "usage": {
                "prompt_tokens": len(text.split()),
                "completion_tokens": len(text.split()) + 1,
                "total_tokens": len(text.split()) * 2 + 1,
            },
        }

    async def generate_stream(self, prompt, **kwargs):
        result = await self.generate(prompt, **kwargs)
        for word in result["generated_text"].split():
            yield word + " "

    async def list_models(self):
        return [
            {"id": "my-model", "object": "model", "created": 0, "owned_by": "me"}
        ]

    async def get_model(self, model_id):
        if model_id == "my-model":
            models = await self.list_models()
            return models[0]
        return None


if __name__ == "__main__":
    openai_http.setup_logging()
    openai_http.run_server(backend=MyBackend(), skip_validation=True)
