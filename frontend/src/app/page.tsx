import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function HomePage() {
  return (
    <main className="p-10 space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Job Analyzer</h1>
        <p className="text-gray-600 mt-2">
          Consumer and enterprise job intelligence platform.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Consumer</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Badge>Job Seeker</Badge>
            <p className="text-sm text-gray-600">
              Analyze job descriptions, detect bias, and find roles that fit
              your skills.
            </p>
            <Link
              href="/consumer"
              className="inline-block text-sm underline text-blue-600"
            >
              Go to Consumer →
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Corporate</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Badge>Employer</Badge>
            <p className="text-sm text-gray-600">
              Define a role, generate a job description, and audit it for
              clarity and bias.
            </p>
            <Link
              href="/corporate"
              className="inline-block text-sm underline text-blue-600"
            >
              Go to Corporate →
            </Link>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
