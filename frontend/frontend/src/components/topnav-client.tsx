"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Badge } from "@/components/ui/badge";

function NavLink({ href, label }: { href: string; label: string }) {
  const path = usePathname();
  const active = path === href || path.startsWith(href + "/");

  return (
    <Link
      href={href}
      className={[
        "text-sm px-3 py-2 rounded-xl transition",
        active ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/70 hover:text-foreground",
      ].join(" ")}
    >
      {label}
    </Link>
  );
}

export function TopNavClient() {
  return (
    <div className="flex items-center gap-2">
      <NavLink href="/consumer" label="Consumer" />
      <NavLink href="/corporate" label="Corporate" />
      <Badge variant="secondary" className="ml-2">Local</Badge>
    </div>
  );
}
