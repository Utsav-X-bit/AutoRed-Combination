import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type ThemeMode = 'light' | 'dark' | 'system';

interface ThemeState {
  mode: ThemeMode;
  resolved: 'light' | 'dark';
  setMode: (mode: ThemeMode) => void;
  toggle: () => void;
}

const getSystemTheme = () =>
  window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      mode: 'system',
      resolved: getSystemTheme(),
      setMode: (mode) => {
        const resolved = mode === 'system' ? getSystemTheme() : mode;
        set({ mode, resolved });
      },
      toggle: () => {
        const nextMode = get().resolved === 'light' ? 'dark' : 'light';
        set({ mode: nextMode, resolved: nextMode });
      },
    }),
    {
      name: 'autored-theme',
      partialize: (state) => ({ mode: state.mode }),
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        const resolved = state.mode === 'system' ? getSystemTheme() : state.mode;
        state.resolved = resolved;
      },
    },
  ),
);
