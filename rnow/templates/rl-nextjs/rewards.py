"""
Reward functions for Next.js ast-grep rules using ReinforceNow framework.
Each reward function checks if the generated code matches the expected ast-grep pattern.
"""

import re

from ast_grep_py import Config, SgRoot

from rnow.core import RewardArgs, reward


@reward(precondition=True)
def code_block(args: RewardArgs, messages: list) -> float:
    """Precondition: Response must contain a ```typescript/tsx code block."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    return 1.0 if match else 0.0


@reward
def layout_syntax_1(args: RewardArgs, messages: list) -> float:
    """Reward for correct Next.js layout syntax with children prop."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(
                rule={
                    "all": [
                        {
                            "pattern": "function $NAME({ children }: { children: React.ReactNode }) { $$$BODY }"
                        },
                        {"kind": "function_declaration"},
                        {"has": {"pattern": "children", "stopBy": "end"}},
                    ]
                }
            )
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def server_dynamic_segment_1(args: RewardArgs, messages: list) -> float:
    """Reward for correct async param extraction in dynamic segment pages."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(
                rule={
                    "all": [
                        {"pattern": "function $FUNC($$$ARGS) { $$$BODY }"},
                        {"kind": "function_declaration"},
                        {"has": {"pattern": "const { $VAR2 } = await params", "stopBy": "end"}},
                    ]
                }
            )
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def server_dynamic_segment_2(args: RewardArgs, messages: list) -> float:
    """Reward for generateStaticParams pattern."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(
                rule={
                    "kind": "program",
                    "all": [
                        {
                            "has": {
                                "pattern": "function generateStaticParams() { $$$BODY }",
                                "has": {
                                    "pattern": "return posts.map((post) => ({ slug: post.slug, }))",
                                    "stopBy": "end",
                                },
                                "stopBy": "end",
                            }
                        }
                    ],
                }
            )
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def server_search_params(args: RewardArgs, messages: list) -> float:
    """Reward for correct server searchParams handling."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(
                rule={
                    "all": [
                        {
                            "pattern": "async function $FUNC({ searchParams }: { searchParams: Promise<{ [key: string]: string | string[] | undefined }> }) { $$$BODY }"
                        },
                        {"kind": "function_declaration"},
                        {
                            "has": {
                                "pattern": "const $VAR = (await searchParams).$VAR",
                                "stopBy": "end",
                            }
                        },
                    ]
                }
            )
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def use_client_directive(args: RewardArgs, messages: list) -> float:
    """Reward for correct 'use client' directive placement."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(Config(rule={"kind": "string", "pattern": '"use client"'}))
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def metadata_export(args: RewardArgs, messages: list) -> float:
    """Reward for valid metadata export."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(Config(rule={"pattern": "export const metadata = { $$$BODY }"}))
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def error_boundary(args: RewardArgs, messages: list) -> float:
    """Reward for valid error boundary component."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(
                rule={
                    "all": [
                        {
                            "pattern": """export default function Error({
  error,
  reset,
}: {
  error: Error
  reset: () => void
}) {
  $$$BODY
}"""
                        },
                        {"has": {"pattern": "reset()", "stopBy": "end"}},
                    ]
                }
            )
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def not_found_boundary(args: RewardArgs, messages: list) -> float:
    """Reward for not-found boundary component."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(rule={"pattern": "export default function NotFound() { $$$BODY }"})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def loading_boundary(args: RewardArgs, messages: list) -> float:
    """Reward for loading boundary component."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(rule={"pattern": "export default function Loading() { $$$BODY }"})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def template_component(args: RewardArgs, messages: list) -> float:
    """Reward for template component."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(
                rule={
                    "all": [
                        {
                            "pattern": "export default function Template({ children }: { children: React.ReactNode }) { $$$BODY }"
                        },
                        {"has": {"pattern": "children", "stopBy": "end"}},
                    ]
                }
            )
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def redirect_usage(args: RewardArgs, messages: list) -> float:
    """Reward for usage of Next.js redirect() helper."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(rule={"kind": "program", "has": {"pattern": "redirect(", "stopBy": "end"}})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def notfound_function_usage(args: RewardArgs, messages: list) -> float:
    """Reward for usage of Next.js notFound() function."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(rule={"kind": "program", "has": {"pattern": "notFound(", "stopBy": "end"}})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def generate_metadata_function(args: RewardArgs, messages: list) -> float:
    """Reward for dynamic generateMetadata() function."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(rule={"pattern": "export async function generateMetadata() { $$$BODY }"})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def generate_metadata_object(args: RewardArgs, messages: list) -> float:
    """Reward for static metadata object export."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(Config(rule={"pattern": "export const metadata = { $$$BODY }"}))
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def route_handler_get(args: RewardArgs, messages: list) -> float:
    """Reward for GET route handler in Next.js Route Handlers."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(rule={"pattern": "export async function GET(request: Request) { $$$BODY }"})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def route_handler_post(args: RewardArgs, messages: list) -> float:
    """Reward for POST route handler in Next.js Route Handlers."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(rule={"pattern": "export async function POST(request: Request) { $$$BODY }"})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def default_page_component(args: RewardArgs, messages: list) -> float:
    """Reward for default page component."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(rule={"pattern": "export default function Page() { $$$BODY }"})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def client_component_detection(args: RewardArgs, messages: list) -> float:
    """Reward for client components using 'use client' directive."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(rule={"all": [{"kind": "string"}, {"pattern": '"use client"'}]})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
def server_component_detection(args: RewardArgs, messages: list) -> float:
    """Reward for server components (without 'use client')."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        has_function = root.find_all(Config(rule={"kind": "function_declaration"}))
        has_use_client = root.find_all(Config(rule={"kind": "string", "pattern": '"use client"'}))
        return 1.0 if has_function and not has_use_client else 0.0
    except Exception:
        return 0.0


@reward
def parallel_route_segment(args: RewardArgs, messages: list) -> float:
    """Reward for parallel route segments (e.g., @modal)."""
    response = messages[-1].get("content", "")
    match = re.search(r"```(?:typescript|tsx|ts)\n(.*?)```", response, re.DOTALL)
    if not match:
        return 0.0

    try:
        root = SgRoot(match.group(1).strip(), "tsx").root()
        matches = root.find_all(
            Config(rule={"kind": "program", "has": {"pattern": "@", "stopBy": "end"}})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0
