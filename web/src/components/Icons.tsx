interface IconProps {
  size?: number;
  className?: string;
}

const base = (size: number, className?: string) => ({
  className,
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
});

export function Logo({ size = 18, className }: IconProps) {
  return (
    <svg {...base(size, className)} stroke="#060a14" strokeWidth={2.2} aria-hidden="true">
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5Z" />
      <circle cx="11.5" cy="12.5" r="3" />
      <path d="m14 15 2.5 2.5" />
    </svg>
  );
}

export function Shield({ size = 13, className }: IconProps) {
  return (
    <svg {...base(size, className)} aria-hidden="true">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
      <path d="M9 12h6" />
    </svg>
  );
}

export function Cpu({ size = 13, className }: IconProps) {
  return (
    <svg {...base(size, className)} aria-hidden="true">
      <rect x="5" y="5" width="14" height="14" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" />
    </svg>
  );
}

export function Upload({ size = 22, className }: IconProps) {
  return (
    <svg {...base(size, className)} strokeWidth={1.6} aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M7 10l5-5 5 5" />
      <path d="M12 5v13" />
    </svg>
  );
}

export function Alert({ size = 15, className }: IconProps) {
  return (
    <svg {...base(size, className)} aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 8v5M12 16h.01" />
    </svg>
  );
}

export function Check({ size = 13, className }: IconProps) {
  return (
    <svg {...base(size, className)} aria-hidden="true">
      <path d="m20 6-11 11-5-5" />
    </svg>
  );
}

export function ArrowLeft({ size = 14, className }: IconProps) {
  return (
    <svg {...base(size, className)} aria-hidden="true">
      <path d="M19 12H5M12 19l-7-7 7-7" />
    </svg>
  );
}

export function External({ size = 12, className }: IconProps) {
  return (
    <svg {...base(size, className)} aria-hidden="true">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <path d="M15 3h6v6M10 14 21 3" />
    </svg>
  );
}

export function initialsOf(label: string): string {
  const words = label.replace(/[^\p{L}\p{N}\s-]/gu, " ").trim().split(/[\s-]+/);
  if (words.length === 0 || !words[0]) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0]! + words[words.length - 1]![0]!).toUpperCase();
}
