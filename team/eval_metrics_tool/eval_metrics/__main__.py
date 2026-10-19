import argparse
import sys

from . import PDFReadError, experiments_section, extract_paper, llm_metrics, rank_metrics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eval_metrics",
                                description="List the main evaluation metrics a paper reports.")
    p.add_argument("pdf")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--model", help="LiteLLM model string, e.g. ollama/qwen3:4b or gpt-4o-mini "
                                   "(default: $EVAL_METRICS_MODEL)")
    p.add_argument("--api-base", help="endpoint URL, e.g. a remote Ollama or OpenAI-compatible "
                                      "server (default: $EVAL_METRICS_API_BASE)")
    p.add_argument("--api-key", help="API key; prefer the provider's env var (OPENAI_API_KEY, ...)")
    p.add_argument("--no-llm", action="store_true", help="use only the dictionary ranking")
    p.add_argument("--show-section", action="store_true",
                   help="print the Experiments section (what the LLM is sent) and exit")
    p.add_argument("--verbose", action="store_true",
                   help="dictionary ranking: also print each metric's score and where it was found")
    args = p.parse_args(argv)
    if args.top_k < 1:
        p.error("--top-k must be >= 1")
    try:
        content = extract_paper(args.pdf)
    except (FileNotFoundError, PDFReadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.show_section:
        print(experiments_section(content))
        return 0
    if not args.no_llm:
        names, reason = llm_metrics(content, args.top_k, args.model, args.api_base,
                                    args.api_key)
        if names is not None:
            for name in names:
                print(name)
            return 0
        print(f"note: {reason}", file=sys.stderr)
    for s in rank_metrics(content)[: args.top_k]:
        if args.verbose:
            where = ", ".join(f"{loc} x{n}" for loc, n in s.where.most_common())
            print(f"{s.name}\tscore={s.score}\t{where}\te.g. {s.example!r}")
        else:
            print(s.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
