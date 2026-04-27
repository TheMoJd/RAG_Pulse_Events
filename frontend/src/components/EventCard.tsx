import type { Source } from "../api";

type Props = {
  source: Source;
};

export default function EventCard({ source }: Props) {
  const { title, url, daterange, location_name, image, score } = source;

  return (
    <a
      href={url ?? undefined}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex gap-3 p-3 bg-white rounded-xl border border-slate-200 hover:border-brand-400 hover:shadow-md transition-all"
    >
      {image && (
        <img
          src={image}
          alt=""
          loading="lazy"
          className="w-20 h-20 object-cover rounded-lg flex-shrink-0"
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).style.display = "none";
          }}
        />
      )}
      <div className="flex-1 min-w-0">
        <h3 className="font-semibold text-sm text-slate-900 group-hover:text-brand-700 line-clamp-2">
          {title ?? "Événement"}
        </h3>
        {daterange && (
          <p className="text-xs text-slate-600 mt-1 flex items-center gap-1">
            <span aria-hidden>📅</span>
            <span className="truncate">{daterange}</span>
          </p>
        )}
        {location_name && (
          <p className="text-xs text-slate-600 flex items-center gap-1">
            <span aria-hidden>📍</span>
            <span className="truncate">{location_name}</span>
          </p>
        )}
        <span className="inline-block mt-2 text-[10px] uppercase tracking-wide font-medium text-brand-600 bg-brand-50 px-2 py-0.5 rounded-full">
          {Math.round(score)}% pertinent
        </span>
      </div>
    </a>
  );
}
