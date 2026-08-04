import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export default class AppErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('AutoRed UI render error:', error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center bg-[var(--surface)] p-6">
          <section className="max-w-lg rounded-xl border border-rose-200 bg-white p-6 shadow-sm dark:border-rose-900/50 dark:bg-stone-900">
            <h1 className="font-display text-lg font-semibold text-rose-700 dark:text-rose-400">Unable to render this run</h1>
            <p className="mt-2 text-sm text-stone-600 dark:text-stone-400">
              The result file contains data the UI could not display.
            </p>
            <pre className="mt-4 max-h-48 overflow-auto rounded-lg bg-rose-50 p-3 text-xs text-rose-800 dark:bg-rose-950/30 dark:text-rose-300">
              {this.state.error.message}
            </pre>
            <a href="/runs" className="mt-4 inline-block text-sm font-medium text-teal-700 hover:underline dark:text-teal-400">
              Back to runs
            </a>
          </section>
        </main>
      );
    }
    return this.props.children;
  }
}
