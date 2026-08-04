import { Link, useLocation, useNavigate } from 'react-router-dom';
import { ThemeToggle } from './ThemeToggle';

const navLinks = [
  { path: '/runs', label: 'Runs' },
  { path: '/benchmarks', label: 'Benchmarks' },
];

export function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const isBenchmark = location.pathname.startsWith('/benchmark');

  return (
    <header className="sticky top-0 z-40 border-b border-stone-200 bg-white/90 backdrop-blur-md dark:border-stone-800 dark:bg-stone-950/90">
      <div className="mx-auto flex h-14 max-w-[1600px] items-center justify-between px-4 lg:px-6">
        <div className="flex items-center gap-6">
          <Link to="/" className="flex items-center gap-2.5">
            <span className="flex h-7 w-7 items-center justify-center rounded-md bg-teal-600 text-white">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </span>
            <span className="font-display text-lg font-semibold tracking-tight text-stone-900 dark:text-stone-100">
              AutoRed
            </span>
          </Link>

          <nav className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => {
              const active = (link.path === '/runs' && location.pathname === '/runs') ||
                (link.path === '/benchmarks' && isBenchmark);
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                    active
                      ? 'bg-stone-100 text-stone-900 dark:bg-stone-800 dark:text-stone-100'
                      : 'text-stone-600 hover:bg-stone-50 hover:text-stone-900 dark:text-stone-400 dark:hover:bg-stone-900 dark:hover:text-stone-100'
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate(isBenchmark ? '/runs' : '/benchmarks')}
            className="hidden sm:inline-flex rounded-lg bg-stone-900 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800 dark:bg-teal-600 dark:hover:bg-teal-500"
          >
            {isBenchmark ? 'Run History' : 'Benchmarks'}
          </button>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
