import json
import os
import logging
from openai import OpenAI
from pydantic import BaseModel
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from dotenv import load_dotenv

# Set up the logging configuration
logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables from .env file
load_dotenv()

LINEAR_API_KEY = os.getenv("LINEAR_API_KEY")
LINEAR_TEAM_NAME = os.getenv("LINEAR_TEAM_NAME")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not LINEAR_API_KEY or not DEEPSEEK_API_KEY:
    raise ValueError("Please set LINEAR_API_KEY and DEEPSEEK_API_KEY in your environment variables.")

if not LINEAR_TEAM_NAME:
    logger.warning("LINEAR_TEAM_NAME environment variable not set. Will ask for it later. More convenient to set it in .env file.")


# Set up the GraphQL client
transport = RequestsHTTPTransport(
    url="https://api.linear.app/graphql",
    headers={"Authorization": f"{LINEAR_API_KEY}"},
    verify=True,
    retries=3,
)

linearClient = Client(transport=transport, fetch_schema_from_transport=True)

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

class LinearIssue(BaseModel):
    title: str
    description: str
    priority: int = 0


def get_file_paths(directory, exclude_file_types=None, exclude_dirs=None, exclude_files=None):
    """
    Recursively get a list of all file paths in a directory, excluding specified file types, directories, and files.

    :param directory: The root directory to start the search
    :param exclude_file_types: A list of file extensions to exclude (e.g., ['.txt', '.log'])
    :param exclude_dirs: A list of directory names to exclude (e.g., ['node_modules', '__pycache__'])
    :param exclude_files: A list of specific filenames to exclude (e.g., ['.env', '.gitignore'])
    :return: A list of file paths
    """
    exclude_file_types = exclude_file_types or []
    exclude_dirs = exclude_dirs or []
    exclude_files = exclude_files or []
    file_paths = []

    for root, dirs, files in os.walk(directory):
        # Remove excluded directories from the traversal
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            # Skip files with excluded extensions or filenames
            if (any(file.endswith(ext) for ext in exclude_file_types) or
                    file in exclude_files):
                continue

            file_paths.append(os.path.join(root, file))

    return file_paths


def get_directory_graph(directory, prefix="", exclude_dirs=None, exclude_files=None):
    """
    Recursively generate a tree-like graph of the directory structure, excluding specified directories and files.

    :param directory: The root directory to start the traversal
    :param prefix: The prefix for the current level of the tree
    :param exclude_dirs: A list of directory names to exclude (e.g., ['node_modules', '__pycache__'])
    :param exclude_files: A list of specific filenames to exclude (e.g., ['.env', '.gitignore'])
    :return: A string representing the directory structure
    """
    exclude_dirs = exclude_dirs or []
    exclude_files = exclude_files or []
    graph = []

    try:
        entries = sorted(os.listdir(directory))  # Sort entries for consistent output
    except (PermissionError, FileNotFoundError):
        return ""

    # Filter out excluded items
    entries = [entry for entry in entries
               if (os.path.isdir(os.path.join(directory, entry)) and entry not in exclude_dirs) or
               (os.path.isfile(os.path.join(directory, entry)) and entry not in exclude_files)]

    for index, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        connector = "└── " if index == len(entries) - 1 else "├── "
        graph.append(f"{prefix}{connector}{entry}")

        if os.path.isdir(path):
            extension = "    " if index == len(entries) - 1 else "│   "
            subgraph = get_directory_graph(path, prefix + extension, exclude_dirs, exclude_files)
            if subgraph:
                graph.append(subgraph)

    return "\n".join(graph)


def analyze_file(file_path, directory_graph):
    file_content = ""
    with open(file_path, 'r') as file:
        file_content = file.read()

    file_name = os.path.basename(file_path)

    prompt = f"""
            # Python Code Analysis - Senior Engineer Evaluation

            You are a senior software engineer tasked with performing a comprehensive code review of a Python file. Your goal is to evaluate the code quality, identify issues, and provide actionable feedback to improve the code. You will consider various aspects including code style, architecture, security, performance, and best practices.

            ## Input Format

            You'll be provided with:
            1. A Python file's content
            2. The filename
            3. The project directory structure

            ## Analysis Criteria

            Analyze the code based on the following criteria:

            ### 1. Code Structure and Organization
            - Evaluate overall code structure and organization
            - Check modularity and single responsibility principle
            - Assess if the file is appropriately placed in the project structure
            - Check import organization and necessity

            ### 2. Code Quality
            - Identify code smells and anti-patterns
            - Evaluate function/method naming and purpose clarity
            - Check variable naming conventions
            - Assess code readability and maintainability
            - Check docstrings and comments quality
            - Evaluate error handling approach

            ### 3. Security Issues
            - Identify hardcoded secrets or credentials
            - Check for insecure API calls
            - Identify potential injection vulnerabilities
            - Check for proper access control
            - Evaluate input validation
            - Assess logging practices (sensitive data exposure)

            ### 4. Performance Considerations
            - Identify potential performance bottlenecks
            - Check for inefficient algorithms or data structures
            - Assess resource management
            - Evaluate concurrency and threading issues
            - Identify repeated operations that could be optimized

            ### 5. Dependency Management
            - Evaluate external library usage
            - Check for deprecated methods/functions
            - Identify potential library version conflicts
            - Assess error handling for external dependencies

            ### 6. Testing Considerations
            - Assess testability of the code
            - Identify areas lacking proper test coverage
            - Check for hardcoded test values

            ### 7. Architecture and Design
            - Evaluate how the file fits into the overall project architecture
            - Check for appropriate abstractions
            - Assess coupling with other modules
            - Identify violations of SOLID principles
            - Evaluate API design (if applicable)

            ### 8. Environment Configuration
            - Assess environment variable handling
            - Check for configuration management issues
            - Identify platform-specific code

            ### 9. Best Practices Compliance
            - Check adherence to Python best practices (PEP 8, etc.)
            - Identify non-Pythonic code patterns
            - Evaluate type hinting usage and correctness
            - Check for proper exception handling

            ### 10. Documentation
            - Assess the completeness of docstrings
            - Check for missing parameter/return documentation
            - Evaluate overall code documentation quality

            ## Output Format

            Provide your analysis in the following format:

            **1. Summary**
            A brief overview of the code and its purpose based on your analysis.

            **2. Critical Issues**
            Identify 3-5 most important issues that should be addressed immediately, ordered by priority.

            **3. Detailed Analysis**
            Organize your detailed findings by the categories listed in the analysis criteria. For each issue:
            - Provide the specific line number or code snippet
            - Explain why it's an issue
            - Suggest a concrete improvement

            **4. Refactoring Suggestions**
            Provide specific code suggestions for the most critical issues. Show both the current code and your suggested improvement.

            **5. Architecture Recommendations**
            Based on the project structure, suggest any architectural improvements that could benefit this code.

            **6. Overall Assessment**
            Provide a high-level assessment of the code quality on a scale of 1-5, with specific justification.

            ## Example Analysis

            Here's a partial example of what your analysis might look like:

            ```
            # Code Analysis: user_authentication.py

            ## Summary
            This file implements user authentication functionality for the application, handling user login, session management, and password validation.

            ## Critical Issues
            1. [HIGH] Hardcoded API secret key at line 42
            2. [HIGH] Passwords are stored in plaintext at line 78
            3. [MEDIUM] No rate limiting for authentication attempts
            4. [MEDIUM] Overly broad exception handling at lines 90-92

            ## Detailed Analysis

            ### Security Issues
            - **Line 42**: Hardcoded API key `API_KEY = "sk_live_12345"` should be moved to environment variables
              ```python
              # Current:
              API_KEY = "sk_live_12345"

              # Suggested:
              API_KEY = os.environ.get("API_KEY")
              if not API_KEY:
                  raise EnvironmentError("API_KEY environment variable is not set")
              ```
            ...
            ```

            Remember to be thorough but concise, focusing on providing actionable feedback that will genuinely improve the code quality.
            """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",
             "content": f"file content: {file_content}, file name: {file_name}, directory graph: {directory_graph}"}
        ],
        stream=False
    )

    return response.choices[0].message.content

def extract_issues(user_prompt):
    system_prompt = """
    The user will provide a code review. Please extract actionable issues and output them in JSON format.

    EXAMPLE INPUT:
    ### 3. Security Issues

    - **Line 45-48**: Hardcoded Sentry DSN should be moved to environment variables
    - **Line 94-99**: Overly permissive CORS configuration (`allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`) is a security risk
    - **Line 67-70**: While environment variables are checked, their values could potentially be logged if an error occurs

    EXAMPLE JSON OUTPUT:
    {
        "issues": [
            {
                "title": "Move Sentry DSN to environment variables",
                "description": "Currently hardcoded in the code. This should be moved to environment variables for security.",
                "priority": 2
            },
            {
                "title": "Restrict CORS configuration",
                "description": "The current CORS configuration is overly permissive and could lead to security vulnerabilities.",
                "priority": 1
            },
            {
                "title": "Check environment variable values before logging",
                "description": "Ensure that sensitive information is not logged when an error occurs.",
                "priority": 3
            }
        ]
    }

    The field "priority" can be 0 - no priority, 1 - urgent, 2 - high, 3 - medium, and 4 - low priority. If you are not sure, please leave it at 0.
    """

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )

    return response

mutation_create_issue = gql(
    """
    mutation IssueCreate($input: IssueCreateInput!) {
        issueCreate(input: $input) {
            success
            issue {
                id
                title
                url
                team {
                    name
                }
            }
        }
    }
    """
)

def get_team_id(team_name: str) -> str:
    """
    Get the team ID from Linear by team name.

    :param team_name:
    :return: team ID
    """
    query_team = gql(
        """
        query GetTeam($teamName: String!) {
            teams(filter: { name: { eq: $teamName } }) {
                nodes {
                    id
                    name
                }
            }
        }
        """
    )

    team_result = linearClient.execute(query_team, variable_values={"teamName": team_name})
    return team_result["teams"]["nodes"][0]["id"]  # Assumes the team exists

def create_issue(title, description, priority, team_id):
    issue_input = {
        "title": title,
        "description": description,
        "teamId": team_id,
        "priority": priority,
        # "stateId": "state_id_here",  # e.g., "Backlog", "In Progress"
    }
    result = linearClient.execute(
        mutation_create_issue,
        variable_values={"input": issue_input},
    )
    print("Issue created successfully!")
    print(f"URL: {result['issueCreate']['issue']['url']}")


# Example usage:
if __name__ == "__main__":
    # Define exclusions
    exclude_dirs = [".idea", "data", ".venv", ".git", ".ruff_cache", ".pytest_cache", "__pycache__", "migrations",
                    "tests", "docs"]
    exclude_files = [".env", ".DS_Store", "__init__.py", ".gitignore"]
    exclude_types = [".pyc", ".pyo", ".pyd"]

    # Get the directory path from the user
    directory_path = input("Enter the directory path to evaluate: ")

    # Generate the directory graph
    directory_graph = get_directory_graph(
        directory_path,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files
    )

    # Get the list of files
    files = get_file_paths(
        directory_path,
        exclude_file_types=exclude_types,
        exclude_dirs=exclude_dirs,
        exclude_files=exclude_files
    )

    print(directory_graph)
    print("\nFile count:", len(files))

    # Print only files that are not Python files
    non_py_files = [f for f in files if not f.endswith(".py")]
    print("\nNon-Python files:")
    for f in non_py_files:
        print(f)

    if not LINEAR_TEAM_NAME:
        # Ask for the team name
        LINEAR_TEAM_NAME = input("Enter the Linear team name: ")

    # Based on the team name, get the team ID
    team_id = get_team_id(LINEAR_TEAM_NAME)

    print("\nAnalyzing Python files...")

    # Analyze each file
    for file_path in files:
        if file_path.endswith(".py"):
            print("Analyzing file:", file_path)
            analysis = analyze_file(file_path, directory_graph)

            print(analysis)

            print("\nExtracting issues...")
            issues = extract_issues(analysis).choices[0].message.content

            print("\nParsing issues...")
            issues_json = json.loads(issues)["issues"]

            # Loop through the issues
            for issue in issues_json:
                issue_in = LinearIssue(
                    title=issue["title"],
                    description=issue["description"] + "\n\nfile path: " + file_path,
                    priority=issue["priority"]
                )

                print(f"Title: {issue_in.title}")
                print(f"Priority: {issue_in.priority}")
                print(f"Description:\n{issue_in.description}")

                print("Create issue in Linear?")

                if input("y/n: ") == "y":
                    # Create the issue in Linear
                    create_issue(title=issue["title"], description=issue["description"], priority=issue["priority"], team_id=team_id)
                else:
                    continue

