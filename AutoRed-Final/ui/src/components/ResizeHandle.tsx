import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';

interface ResizeHandleProps {
  direction: 'horizontal' | 'vertical';
  label: string;
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onReset: () => void;
  onKeyboardResize: (delta: number) => void;
}

export default function ResizeHandle({
  direction,
  label,
  onPointerDown,
  onReset,
  onKeyboardResize,
}: ResizeHandleProps) {
  const vertical = direction === 'vertical';

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 40 : 12;
    if (vertical && event.key === 'ArrowLeft') {
      event.preventDefault();
      onKeyboardResize(-step);
    } else if (vertical && event.key === 'ArrowRight') {
      event.preventDefault();
      onKeyboardResize(step);
    } else if (!vertical && event.key === 'ArrowUp') {
      event.preventDefault();
      onKeyboardResize(-step);
    } else if (!vertical && event.key === 'ArrowDown') {
      event.preventDefault();
      onKeyboardResize(step);
    } else if (event.key === 'Home') {
      event.preventDefault();
      onReset();
    }
  };

  return (
    <div
      role="separator"
      aria-label={label}
      aria-orientation={vertical ? 'vertical' : 'horizontal'}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onDoubleClick={onReset}
      onKeyDown={handleKeyDown}
      className={`group relative z-20 flex-shrink-0 touch-none select-none outline-none ${
        vertical ? 'w-2 cursor-col-resize' : 'h-2 cursor-row-resize'
      }`}
      title={`${label}. Drag to resize; double-click to reset.`}
    >
      <span
        className={`absolute rounded-full bg-slate-300 transition-all group-hover:bg-blue-500 group-focus:bg-blue-500 ${
          vertical
            ? 'inset-y-0 left-1/2 w-px -translate-x-1/2 group-hover:w-1 group-focus:w-1'
            : 'inset-x-0 top-1/2 h-px -translate-y-1/2 group-hover:h-1 group-focus:h-1'
        }`}
      />
      <span
        className={`absolute rounded-full border border-slate-300 bg-white shadow-sm opacity-0 transition-opacity group-hover:opacity-100 group-focus:opacity-100 ${
          vertical
            ? 'left-1/2 top-1/2 h-10 w-2 -translate-x-1/2 -translate-y-1/2'
            : 'left-1/2 top-1/2 h-2 w-10 -translate-x-1/2 -translate-y-1/2'
        }`}
      />
    </div>
  );
}
