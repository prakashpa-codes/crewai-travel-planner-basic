import os
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Task


def main():
    load_dotenv()  # Load environment variables from .env file if present
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Set it in your PowerShell session before running this script."
        )

    model_name = os.getenv("MODEL", "openai/gpt-4o-mini")
    city = input("Which city? ").strip()

    if not city:
        raise ValueError("Please enter a city name.")

    llm = LLM(model=model_name, api_key=api_key)

    planner = Agent(
        role="Travel Planner",
        goal=(
            "Recommend exactly three worthwhile places to visit in the city provided by the user, "
            "with concise and useful guidance for each place."
        ),
        backstory=(
            "You are an experienced travel planner who helps visitors make practical sightseeing "
            "choices. You prioritize notable attractions and explain briefly what makes each place "
            "worth visiting, while keeping recommendations specific to the requested city."
        ),
        llm=llm,
        verbose=True,
    )

    task = Task(
        description=(
            "Suggest exactly three places to visit in {city}. Choose places that are genuinely "
            "located in or closely associated with that city. For each place, provide its name and "
            "one concise sentence explaining why to visit or a useful visiting tip. Do not include "
            "an introduction, conclusion, extra places, or unrelated travel advice."
        ),
        expected_output=(
            "Exactly three numbered items (1–3). Each item must contain the place name followed "
            "by one concise sentence with a reason to visit or a practical tip. No other sections."
        ),
        agent=planner,
    )

    crew = Crew(agents=[planner], tasks=[task], verbose=True)
    result = crew.kickoff(inputs={"city": city})
    print("\nTravel recommendations:\n")
    print(result)


if __name__ == "__main__":
    main()
