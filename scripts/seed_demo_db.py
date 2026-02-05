import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path("data") / "demo.db"

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()

        cur.execute("""
        CREATE TABLE docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL
        )
        """)

        cur.execute("""
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            ts TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL
        )
        """)

        docs = [
            ("Runbook: Deploy seguro do serviço API", "runbook",
             "1) Verificar healthcheck.\n2) Aplicar migration.\n3) Deploy blue/green.\n4) Validar métricas.\n5) Rollback se erro > 2%."),

            ("KB: Erro 502 no Gateway", "kb",
             "Causas comuns: upstream fora do ar, timeout, DNS.\nAções: checar logs do gateway, validar conexão com upstream, revisar timeouts."),

            ("Runbook: Rotação de credenciais (demo)", "runbook",
             "1) Gerar credencial nova.\n2) Atualizar segredo.\n3) Reiniciar serviço.\n4) Validar acesso.\n5) Revogar credencial antiga."),

            ("KB: Lentidão em consultas no banco", "kb",
             "Verificar índices, plano de execução, cardinalidade e lock.\nMitigações: LIMIT, índices, e reduzir payload."),
        ]
        cur.executemany("INSERT INTO docs (title, category, content) VALUES (?, ?, ?)", docs)

        base = datetime.utcnow() - timedelta(hours=6)
        events = []
        for i in range(30):
            ts = (base + timedelta(minutes=12*i)).isoformat()
            asset = "gateway" if i % 2 == 0 else "worker"
            severity = "INFO"
            msg = "Healthcheck OK" if i % 3 else "Latency spike detected"
            if i % 10 == 0:
                severity = "WARN"
                msg = "Retry rate increased"
            events.append((asset, ts, severity, msg))

        cur.executemany("INSERT INTO events (asset, ts, severity, message) VALUES (?, ?, ?, ?)", events)

        con.commit()
        print(f"Banco demo criado em: {DB_PATH.resolve()}")
    finally:
        con.close()

if __name__ == "__main__":
    main()
