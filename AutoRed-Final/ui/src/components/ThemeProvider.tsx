import { useEffect, type ReactNode } from 'react';
import { useThemeStore } from '../store/themeStore';

export function ThemeProvider({ children }: { children: ReactNode }) {
  const { resolved } = useThemeStore();

  useEffect(() => {
    const root = window.document.documentElement;
    if (resolved === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [resolved]);

  return <>{children}</>;
}
