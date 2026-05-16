import React from "react";

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends React.Component<React.PropsWithChildren, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error(error);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="grid min-h-screen place-items-center bg-slate-950 px-6 text-slate-100">
          <section className="w-full max-w-lg rounded-md border border-red-500/40 bg-red-950/60 p-5">
            <h1 className="text-lg font-semibold">Lumina Learn crashed</h1>
            <p className="mt-2 text-sm text-red-100">{this.state.error.message}</p>
            <button
              type="button"
              className="mt-4 rounded-md border border-red-300/50 px-3 py-2 text-sm text-red-50"
              onClick={() => this.setState({ error: null })}
            >
              Retry
            </button>
          </section>
        </main>
      );
    }
    return this.props.children;
  }
}
