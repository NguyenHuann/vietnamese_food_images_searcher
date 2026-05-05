import { NavLink, Link } from 'react-router-dom';
import { History, Focus } from 'lucide-react';
import { cn } from '@/src/lib/utils';

export function Navigation() {
  return (
    <>
      <header className="bg-surface-container-low/90 backdrop-blur-2xl sticky top-0 z-40 border-b border-primary/10 shadow-[0_4px_30px_rgba(0,0,0,0.03)] transition-all duration-300">
        <div className="flex justify-center items-center w-full px-6 py-3 max-w-7xl mx-auto">
          <nav className="flex items-center gap-2 md:gap-4">
            <NavLink to="/" className={({ isActive }) => cn(
              "text-[10px] md:text-xs font-black uppercase tracking-[0.2em] md:tracking-[0.25em] transition-all duration-300 px-4 md:px-8 py-2 md:py-3 rounded-xl md:rounded-2xl flex items-center gap-2",
              isActive 
                ? "bg-primary text-white shadow-lg shadow-primary/20 scale-105" 
                : "text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface"
            )}>
              Khám phá
            </NavLink>
            <NavLink to="/history" className={({ isActive }) => cn(
              "text-[10px] md:text-xs font-black uppercase tracking-[0.2em] md:tracking-[0.25em] transition-all duration-300 px-4 md:px-8 py-2 md:py-3 rounded-xl md:rounded-2xl flex items-center gap-2",
              isActive 
                ? "bg-primary text-white shadow-lg shadow-primary/20 scale-105" 
                : "text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface"
            )}>
              Lịch sử
            </NavLink>
          </nav>
        </div>
      </header>

      {/* Mobile Bottom Nav */}
      <nav className="md:hidden fixed bottom-0 left-0 w-full z-50 flex justify-around items-center px-4 pb-6 pt-3 bg-white shadow-[0_-8px_30px_rgba(0,0,0,0.05)] border-t border-primary/5">
        <NavLink 
          to="/" 
          className={({ isActive }) => cn(
            "flex flex-col items-center justify-center rounded-2xl px-8 py-2 transition-all duration-500",
            isActive ? "bg-primary text-white shadow-xl shadow-primary/20 -translate-y-1" : "text-on-surface-variant hover:text-primary"
          )}
        >
          {({ isActive }) => (
            <>
              <Focus size={20} className={cn("transition-transform duration-500", isActive && "scale-110")} />
              <span className="text-[10px] font-black uppercase tracking-widest mt-1">Khám phá</span>
            </>
          )}
        </NavLink>

        <NavLink 
          to="/history" 
          className={({ isActive }) => cn(
            "flex flex-col items-center justify-center rounded-2xl px-8 py-2 transition-all duration-500",
            isActive ? "bg-primary text-white shadow-xl shadow-primary/20 -translate-y-1" : "text-on-surface-variant hover:text-primary"
          )}
        >
          {({ isActive }) => (
            <>
              <History size={20} className={cn("transition-transform duration-500", isActive && "scale-110")} />
              <span className="text-[10px] font-black uppercase tracking-widest mt-1">Lịch sử</span>
            </>
          )}
        </NavLink>
      </nav>
    </>
  );
}
