# Bloqueio de reinicio automatico — 2026-07-15 EDT

## Motivo

O System Event Log provou que o incidente da madrugada nao foi queda de energia
nem bugcheck. O Windows Update iniciou tres reinicios planejados:

- `2026-07-15 02:14:03 EDT`: `MoUsoCoreWorker.exe`, atualizacao/service pack;
- `2026-07-15 02:16:59 EDT`: `TrustedInstaller.exe`, atualizacao do sistema;
- `2026-07-15 02:17:35 EDT`: `TrustedInstaller.exe`, atualizacao do sistema.

O ultimo boot ocorreu em `2026-07-15 02:17:57 EDT`. O benchmark noturno tinha
resultado parcial gravado as `02:07:45 EDT` e ficou incompleto.

## Mudancas aplicadas como administrador

Em `HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU`:

- `NoAutoUpdate=1`;
- `NoAutoRebootWithLoggedOnUsers=1`;
- `AUOptions=2`;
- `AlwaysAutoRebootAtScheduledTime=0`.

Em `HKLM\SYSTEM\CurrentControlSet\Control\CrashControl`:

- `AutoReboot=0`;
- dumps permaneceram habilitados (`CrashDumpEnabled=3`).

As tarefas `UpdateOrchestrator` com nome `Reboot` ou `Restart` foram procuradas;
nenhuma existe no build atual. `wuauserv` foi parado e permanece em start manual.
Nao havia shutdown pendente para abortar (`shutdown /a` retornou 1116).

## Validacao

As quatro politicas foram relidas do registro com os valores acima. O reboot
automatico apos falha de sistema foi relido como desativado. Nao havia marcador
`CBS RebootPending` nem `WindowsUpdate RebootRequired`; existe apenas
`PendingFileRenameOperations`, que por si so nao agenda reinicio.

## Limites honestos

O bloqueio cobre reinicio automatico do Windows Update e reinicio automatico
apos tela azul. Nao consegue impedir queda de energia, reset fisico, botao de
forca, travamento de hardware/driver ou comando manual executado por administrador.

Nenhum treino foi iniciado ou encerrado e nenhum checkpoint foi alterado.
