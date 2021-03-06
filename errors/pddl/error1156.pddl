(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b8 t0)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b0 b6)
		(in-tower b0)
		(on b1 b0)
		(in-tower b1)
		(on b2 b1)
		(in-tower b2)
		(on b3 b2)
		(in-tower b3)
		(blue b0)
		(blue b1)
		(red b2)
		(blue b2)
		(red b3)
		(blue b3)
		(red b4)
		(blue b4)
		(red b5)
		(red b6)
		(red b9)
		(blue b8)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b9 t0)) (not (on b9 b8)) (not (on b9 b7)) (not (on b9 b6)) (not (on b4 b3)))))
)