// The text the model writes, one token at a time.

export class StoryPanel {
  private readonly caret = Object.assign(document.createElement("span"), { className: "caret" });
  private finished = false;

  constructor(private readonly element: HTMLElement) {
    element.append(this.caret);
  }

  /**
   * Add a token. It is dimmer the less sure the model was, and amber when
   * `unexpected`: a healthy copy would almost never have written it.
   */
  add(text: string, probability: number, unexpected = false): void {
    if (this.finished) this.clear();
    const span = document.createElement("span");
    span.textContent = text;
    span.style.opacity = (0.5 + 0.5 * probability).toFixed(2);
    if (unexpected) span.className = "unexpected";
    this.caret.before(span);
  }

  /** The story is over. It stays on screen until the next token arrives. */
  finish(): void {
    this.finished = true;
  }

  private clear(): void {
    this.element.replaceChildren(this.caret);
    this.finished = false;
  }
}
