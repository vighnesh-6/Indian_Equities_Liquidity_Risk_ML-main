import { NextRequest, NextResponse } from "next/server";
import { getResource, InputError } from "@/lib/api";

export const dynamic = "force-dynamic";
export const maxDuration = 30;

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ resource: string }> },
) {
  try {
    const { resource } = await context.params;
    const data = await getResource(resource, request.nextUrl.searchParams);
    return NextResponse.json(data, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    if (error instanceof InputError)
      return NextResponse.json(
        { error: "INVALID_REQUEST", message: error.message },
        { status: 400 },
      );
    console.error(
      "Liquidity API request failed",
      error instanceof Error ? error.name : "UnknownError",
      typeof (error as { code?: unknown })?.code === "string" &&
        /^[A-Z0-9_]{5,30}$/.test((error as { code: string }).code)
        ? (error as { code: string }).code
        : "NO_SQLSTATE",
    );
    return NextResponse.json(
      {
        error: "DATA_UNAVAILABLE",
        message:
          "Production data is currently unavailable. Check Data & System Status or retry shortly.",
      },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
