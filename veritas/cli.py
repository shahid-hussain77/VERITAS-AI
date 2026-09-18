"""VERITAS-AI CLI."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from veritas import __version__
from veritas.orchestrator import Orchestrator
from veritas.config import REPORTS_DIR


console = Console()


@click.group()
@click.version_option(__version__)
def cli():
    """VERITAS-AI — Local Multi-Agent Academic Integrity Engine."""
    pass


@cli.command()
@click.argument("submitted", type=click.Path(exists=True))
@click.option("--source", "-s", multiple=True, type=click.Path(exists=True),
              help="Source document(s) to compare against")
@click.option("--history", "-h", multiple=True, type=click.Path(exists=True),
              help="Author's previous submissions")
@click.option("--corpus", "-c", multiple=True, type=click.Path(exists=True),
              help="Local corpus for source discovery")
@click.option("--classroom", multiple=True, type=click.Path(exists=True),
              help="Classroom batch documents")
@click.option("--output", "-o", type=click.Path(),
              help="Output JSON report path")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
def analyze(submitted, source, history, corpus, classroom, output, quiet):
    """Analyze a single submitted document."""
    console.print(Panel.fit(
        f"[bold green]VERITAS-AI v{__version__}[/bold green]\n"
        f"[dim]Analyzing: {Path(submitted).name}[/dim]",
        border_style="green",
    ))

    orch = Orchestrator(verbose=not quiet)
    result = orch.run(
        submitted_path=submitted,
        source_paths=list(source) if source else None,
        author_history_paths=list(history) if history else None,
        corpus_paths=list(corpus) if corpus else None,
        classroom_paths=list(classroom) if classroom else None,
    )

    if result["status"] != "ok":
        console.print(f"[red]Failed: {result.get('error')}[/red]")
        sys.exit(1)

    # Print report summary
    report = result["report"]
    score = report.get("score", {})

    console.print()
    table = Table(title="Integrity Report", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Document", report.get("document_name", "?"))
    table.add_row("Words", str(report.get("document_words", 0)))
    table.add_row("Pages", str(report.get("document_pages", 0)))
    table.add_row("", "")
    table.add_row("Exact overlap", f"{score.get('exact_overlap', 0) * 100:.1f}%")
    table.add_row("Semantic overlap", f"{score.get('semantic_overlap', 0) * 100:.1f}%")
    table.add_row("Paraphrase", f"{score.get('paraphrase_overlap', 0) * 100:.1f}%")
    table.add_row("Cross-language", f"{score.get('cross_language_overlap', 0) * 100:.1f}%")
    table.add_row("Self-overlap", f"{score.get('self_overlap', 0) * 100:.1f}%")
    table.add_row("Source matches", str(score.get("source_matches", 0)))
    table.add_row("Citation issues", str(score.get("citation_issues", 0)))
    table.add_row("Reference issues", str(score.get("reference_issues", 0)))
    table.add_row("AI-writing", str(score.get("ai_writing_signal", "?")))
    table.add_row("Collusion clusters", str(score.get("collusion_clusters", 0)))
    table.add_row("", "")
    table.add_row("Overall confidence", str(score.get("overall_confidence", "?")))
    table.add_row("Human review", "YES" if report.get("human_review_recommended") else "NO")

    console.print(table)

    # Save output
    if output:
        out_path = Path(output)
    else:
        out_path = REPORTS_DIR / f"{Path(submitted).stem}_report.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    console.print(f"\n[green]✓ Report saved:[/green] {out_path}")


@cli.command()
@click.argument("file_a", type=click.Path(exists=True))
@click.argument("file_b", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path())
def compare(file_a, file_b, output):
    """Compare two documents."""
    console.print(f"[cyan]Comparing: {Path(file_a).name} vs {Path(file_b).name}[/cyan]")

    orch = Orchestrator(verbose=True)
    result = orch.run(
        submitted_path=file_a,
        source_paths=[file_b],
    )

    if result["status"] != "ok":
        console.print(f"[red]Failed[/red]")
        sys.exit(1)

    report = result["report"]
    console.print(f"\n[bold]Findings:[/bold] {report.get('findings_count', 0)}")

    if output:
        Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False))


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", type=click.Path())
def batch(folder, output):
    """Run classroom batch analysis on a folder."""
    folder = Path(folder)
    files = list(folder.glob("*.pdf")) + list(folder.glob("*.docx")) + \
            list(folder.glob("*.txt"))

    if len(files) < 2:
        console.print("[red]Need at least 2 documents[/red]")
        sys.exit(1)

    console.print(f"[cyan]Batch analyzing {len(files)} documents[/cyan]")

    # Use first as submitted, rest as classroom
    orch = Orchestrator(verbose=True)
    result = orch.run(
        submitted_path=str(files[0]),
        classroom_paths=[str(f) for f in files[1:]],
    )

    if result["status"] != "ok":
        console.print("[red]Failed[/red]")
        sys.exit(1)

    report = result["report"]
    console.print(f"\n[bold]Classroom Report[/bold]")
    console.print(f"  Collusion clusters: {report['score'].get('collusion_clusters', 0)}")

    if output:
        Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        console.print(f"[green]Saved: {output}[/green]")


@cli.command()
def stats():
    """Show database statistics."""
    from veritas.memory.store import memory
    s = memory.stats()
    console.print("[bold]Database Statistics[/bold]")
    for k, v in s.items():
        console.print(f"  {k}: {v}")


def main():
    cli()


if __name__ == "__main__":
    main()