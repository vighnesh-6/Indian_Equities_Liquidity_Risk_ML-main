import calendarData from './session-calendar.json';

type Calendar = { holidays:string[]; excluded_sessions:Record<string,unknown>; special_sessions:Record<string,{ready_at?:string}>; ready_at:string };

export function latestReadySession(now = new Date()): string | null {
  const parts = new Intl.DateTimeFormat('en-CA', { timeZone:'Asia/Kolkata', year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23' }).formatToParts(now);
  const value = (key:string) => parts.find(part=>part.type===key)?.value || '';
  const today = value('year')+'-'+value('month')+'-'+value('day');
  const clock = value('hour')+':'+value('minute');
  const candidate = new Date(today+'T12:00:00Z');
  for (let offset=0;offset<30;offset++) {
    const day = candidate.toISOString().slice(0,10);
    const year = (calendarData as Record<string,Calendar>)[day.slice(0,4)];
    if (!year) return null;
    const weekday = candidate.getUTCDay();
    const session = !(day in year.excluded_sessions) && (day in year.special_sessions || (weekday!==0 && weekday!==6 && !year.holidays.includes(day)));
    const ready = year.special_sessions[day]?.ready_at || year.ready_at;
    if (session && (day<today || clock>=ready)) return day;
    candidate.setUTCDate(candidate.getUTCDate()-1);
  }
  return null;
}
