import "./globals.css";
import { TopNavClient } from "@/components/topnav-client";
import { Toaster } from "@/components/ui/sonner";

export const metadata = {
  title: "Job Analyzer",
  description: "Consumer + Corporate job intelligence",
};

function SideLink({
  href,
  title,
  subtitle,
}: {
  href: string;
  title: string;
  subtitle: string;
}) {
  return (
    <a
      href={href}
      className="block rounded-2xl px-4 py-3 border border-transparent hover:border-gray-200 hover:bg-gray-50 transition"
    >
      <div className="text-sm font-medium">{title}</div>
      <div className="text-xs text-gray-500 mt-1">{subtitle}</div>
    </a>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900">
        <div className="min-h-screen">
          {/* Top Bar */}
          <header className="bg-white border-b">
            <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-6">
              <a href="/" className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-2xl bg-gray-900 shadow-sm" />
                <div className="leading-tight">
                  <div className="font-semibold tracking-tight">
                    Job Analyzer
                  </div>
                  <div className="text-xs text-gray-500">
                    Enterprise UI preview
                  </div>
                </div>
              </a>

              <TopNavClient />
            </div>
          </header>

          {/* Main */}
          <div className="max-w-7xl mx-auto px-6 py-8">
            <div className="grid grid-cols-12 gap-6">
              <aside className="col-span-12 md:col-span-4 lg:col-span-3">
                <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-3 sticky top-6">
                  <div className="px-3 pt-3 pb-2 text-xs font-semibold text-gray-500">
                    Workspaces
                  </div>

                  <SideLink
                    href="/consumer"
                    title="Consumer"
                    subtitle="Search + screen job descriptions"
                  />
                  <SideLink
                    href="/corporate"
                    title="Corporate"
                    subtitle="Audit, rewrite, generate roles"
                  />
                </div>
              </aside>

              <main className="col-span-12 md:col-span-8 lg:col-span-9">
                {children}
              </main>
            </div>
          </div>

          <footer className="bg-white border-t">
            <div className="max-w-7xl mx-auto px-6 py-4 text-xs text-gray-500">
              Prototype UI, production-style shell
            </div>
          </footer>
        </div>

        <Toaster />
      </body>
    </html>
  );
}


