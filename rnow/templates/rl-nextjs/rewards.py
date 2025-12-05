"""
Reward functions for Next.js ast-grep rules using ReinforceNow framework.
Each reward function checks if the generated code matches the expected ast-grep pattern.
"""

from ast_grep_py import Config, SgRoot

from rnow.core import RewardArgs, reward


def _extract_first_code_block(text: str) -> str:
    """
    Return the contents of the first ```typescript or ```tsx code block in text.
    If not found, try <code>...</code> blocks.
    If neither found, return the text as-is.
    """
    if not isinstance(text, str):
        return ""

    # Try markdown code blocks first
    for marker in ["```typescript", "```tsx", "```ts", "```"]:
        start = text.find(marker)
        if start != -1:
            start += len(marker)
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

    # Fallback to <code> tags
    start = text.find("<code>")
    if start != -1:
        end = text.find("</code>", start + 6)
        if end != -1:
            return text[start + 6 : end].strip()

    # Return the whole text if no markers found
    return text.strip()


@reward
async def layout_syntax_1(args: RewardArgs, messages: list) -> float:
    """
    Reward for correct Next.js layout syntax with children prop.
    Reference: layout-syntax-1.yml
    """
    response = messages[-1].get("content", "") if messages else ""
    tsx_code = _extract_first_code_block(response)

    if not tsx_code:
        return 0.0

    try:
        root = SgRoot(tsx_code, "tsx").root()
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
async def server_dynamic_segment_1(args: RewardArgs, messages: list) -> float:
    """
    Reward for correct async param extraction in dynamic segment pages.
    Reference: server-dynamic-segment-1.yml
    """
    response = messages[-1].get("content", "") if messages else ""
    tsx_code = _extract_first_code_block(response)

    if not tsx_code:
        return 0.0

    try:
        root = SgRoot(tsx_code, "tsx").root()
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
async def server_dynamic_segment_2(args: RewardArgs, messages: list) -> float:
    """
    Reward for generateStaticParams pattern.
    Reference: server-dynamic-segment-2.yml
    """
    response = messages[-1].get("content", "") if messages else ""
    tsx_code = _extract_first_code_block(response)

    if not tsx_code:
        return 0.0

    try:
        root = SgRoot(tsx_code, "tsx").root()
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
async def server_search_params(args: RewardArgs, messages: list) -> float:
    """
    Reward for correct server searchParams handling.
    Reference: server-search-params.yml
    """
    response = messages[-1].get("content", "") if messages else ""
    tsx_code = _extract_first_code_block(response)

    if not tsx_code:
        return 0.0

    try:
        root = SgRoot(tsx_code, "tsx").root()
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
async def use_client_directive(args: RewardArgs, messages: list) -> float:
    """
    Reward for correct 'use client' directive placement.
    Reference: use-client-directive.yml
    """
    response = messages[-1].get("content", "") if messages else ""
    tsx_code = _extract_first_code_block(response)

    if not tsx_code:
        return 0.0

    try:
        root = SgRoot(tsx_code, "tsx").root()
        matches = root.find_all(Config(rule={"kind": "string", "pattern": '"use client"'}))
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
async def metadata_export(args: RewardArgs, messages: list) -> float:
    """
    Reward for valid metadata export.
    Reference: metadata-export.yml
    """
    response = messages[-1].get("content", "") if messages else ""
    tsx_code = _extract_first_code_block(response)

    if not tsx_code:
        return 0.0

    try:
        root = SgRoot(tsx_code, "tsx").root()
        matches = root.find_all(Config(rule={"pattern": "export const metadata = { $$$BODY }"}))
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
async def error_boundary(args: RewardArgs, messages: list) -> float:
    """
    Reward for valid error boundary component.
    Reference: error-boundary.yml
    """
    response = messages[-1].get("content", "") if messages else ""
    tsx_code = _extract_first_code_block(response)

    if not tsx_code:
        return 0.0

    try:
        root = SgRoot(tsx_code, "tsx").root()
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
async def not_found_boundary(args: RewardArgs, messages: list) -> float:
    """
    Reward for not-found boundary component.
    Reference: not-found-boundary.yml
    """
    response = messages[-1].get("content", "") if messages else ""
    tsx_code = _extract_first_code_block(response)

    if not tsx_code:
        return 0.0

    try:
        root = SgRoot(tsx_code, "tsx").root()
        matches = root.find_all(
            Config(rule={"pattern": "export default function NotFound() { $$$BODY }"})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
async def loading_boundary(args: RewardArgs, messages: list) -> float:
    """
    Reward for loading boundary component.
    Reference: loading-boundary.yml
    """
    response = messages[-1].get("content", "") if messages else ""
    tsx_code = _extract_first_code_block(response)

    if not tsx_code:
        return 0.0

    try:
        root = SgRoot(tsx_code, "tsx").root()
        matches = root.find_all(
            Config(rule={"pattern": "export default function Loading() { $$$BODY }"})
        )
        return 1.0 if matches else 0.0
    except Exception:
        return 0.0


@reward
async def template_component(args: RewardArgs, messages: list) -> float:
    """
    Reward for template component.
    Reference: template-component.yml
    """
    response = messages[-1].get("content", "") if messages else ""
    tsx_code = _extract_first_code_block(response)

    if not tsx_code:
        return 0.0

    try:
        root = SgRoot(tsx_code, "tsx").root()
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
