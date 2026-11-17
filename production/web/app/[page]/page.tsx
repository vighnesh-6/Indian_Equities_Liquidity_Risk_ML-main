import { notFound } from "next/navigation";
import Dashboard from "@/components/dashboard";

const pages = ["screener", "stock", "anomalies", "events", "status"];

export default async function Page({
  params,
}: {
  params: Promise<{ page: string }>;
}) {
  const { page } = await params;
  if (!pages.includes(page)) notFound();
  return <Dashboard key={page} page={page} />;
}
