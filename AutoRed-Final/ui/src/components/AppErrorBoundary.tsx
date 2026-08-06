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
        <main className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
          <section className="max-w-lg rounded-xl border border-red-200 bg-white p-6 shadow-sm">
            <h1 className="text-lg font-bold text-red-700">Unable to render this run</h1>
            <p className="mt-2 text-sm text-slate-600">
              The result file contains data the UI could not display.
            </p>
            <pre className="mt-4 overflow-auto rounded-lg bg-red-50 p-3 text-xs text-red-800">
              {this.state.error.message}
            </pre>
            <a href="/runs" className="mt-4 inline-block text-sm font-medium text-blue-700">
              Back to runs
            </a>
          </section>
        </main>
      );
    }
    return this.props.children;
  }
}
